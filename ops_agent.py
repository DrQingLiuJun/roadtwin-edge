#!/usr/bin/env python3
"""Forced SSH command: inspect or control only the vision service."""
import fcntl,json,os,re,subprocess,time,shutil,tempfile,sys
sys.dont_write_bytecode=True
from pathlib import Path
import edge_tracker_config as tracker_config
UNIT='rsu-edge-vision.service'
BASE=Path('/home/nvidia/bushuchengxu')
MODELS=BASE/'models'
SELECTION=BASE/'selected_model.json'
MODEL_ID=r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}'
SERVICE_CMD=['/usr/bin/sudo','-n','/usr/bin/systemctl']
def run(args,timeout=4):
    p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr.strip()[:180] or 'Command failed')
    return p.stdout

def camera_metrics(records, invocation, now):
    """Only fresh frames from the current service invocation; zero is valid."""
    fps,counts={},{}
    if not invocation:return fps,counts
    pattern=r'\[Cam (\d+)\] fps=([\d.]+) infer_ms=([\d.]+) queue_ms=([\d.]+).*? results=(\d+)'
    for line in records.splitlines():
        try:
            row=json.loads(line)
            if row.get('_SYSTEMD_INVOCATION_ID')!=invocation:continue
            stamp=int(row['__REALTIME_TIMESTAMP'])/1e6
            age=now-stamp
            if not 0<=age<=4:continue
            match=re.search(pattern,row.get('MESSAGE',''))
            if not match:continue
            cam,f,i,q,n=match.groups()
            if cam in counts and stamp<=counts[cam]['observed_at']:continue
            fps[cam]={'fps':float(f),'infer_ms':float(i),'queue_ms':float(q)}
            counts[cam]={'objects':int(n),'observed_at':stamp,'age_seconds':age}
        except (ValueError,TypeError,KeyError):continue
    return fps,counts

def cpu_counters():
    values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(values),values[3]+values[4]

def cpu_percent():
    try:
        total,idle=cpu_counters()
        time.sleep(.2)
        new_total,new_idle=cpu_counters()
        delta=new_total-total
        return round(max(0,min(100,100*(1-(new_idle-idle)/delta))),1) if delta>0 else None
    except (OSError,ValueError,IndexError):return None

def gpu_percent():
    for path in ('/sys/devices/platform/bus@0/17000000.gpu/load','/sys/devices/platform/bus@0/gpu.0/load'):
        try:
            load=int(Path(path).read_text())
            if 0<=load<=1000:return load/10
        except (OSError,ValueError):pass
    return None

def model_catalog():
    catalog={}
    for manifest in sorted(MODELS.glob('*/selection.json')):
        try:
            if manifest.is_symlink() or manifest.parent.is_symlink():continue
            cfg=json.loads(manifest.read_text())
            if not isinstance(cfg,dict) or not re.fullmatch(MODEL_ID,cfg['name']):continue
            if cfg['library']!='runtime/libdesktop_detector.so':continue
            paths=[(BASE/cfg[key]).resolve() for key in ('pt','engine')]
            if not all(p.is_relative_to(MODELS) and p.is_file() for p in paths):continue
            if [p.suffix for p in paths]!=['.pt','.engine']:continue
            catalog[cfg['name']]=cfg
        except (OSError,ValueError,KeyError,TypeError):continue
    return catalog

def write_selection(data):
    with tempfile.NamedTemporaryFile(dir=BASE,prefix='.selection-',delete=False) as f:
        temporary=Path(f.name)
        try:
            f.write(data)
            f.close()
            temporary.replace(SELECTION)
        finally:temporary.unlink(missing_ok=True)

def activate_model(model):
    cfg=model_catalog().get(model)
    if cfg is None:raise ValueError('模型未注册或缺少运行文件')
    previous=SELECTION.read_bytes()
    previous_active=run(['/usr/bin/systemctl','show',UNIT,'-p','ActiveState','--value']).strip()=='active'
    if json.loads(previous)==cfg and previous_active:
        state=status()
        if state['cameras'] and set(map(str,state['cameras']))<=state['fps'].keys():return
    restarted=False
    try:
        write_selection(json.dumps(cfg,indent=2).encode()+b'\n')
        run(['/usr/bin/python3','-B',str(BASE/'fenfa_gps.py'),'--check'],12)
        restarted=True
        run([*SERVICE_CMD,'restart',UNIT],16)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            state=status()
            if state['unit'].get('ActiveState')=='active' and state['cameras'] and set(map(str,state['cameras']))<=state['fps'].keys():return
            time.sleep(.35)
        raise RuntimeError('切换后未收到所有相机的检测输出')
    except Exception as error:
        write_selection(previous)
        if restarted:run([*SERVICE_CMD,'restart' if previous_active else 'stop',UNIT],12)
        raise RuntimeError(f'模型切换失败，已恢复原设置：{error}') from error

def apply_tracker(command):
    request=json.loads(bytes.fromhex(command))
    current=tracker_config.load()
    if request.get('revision')!=tracker_config.revision(current):
        raise ValueError('ByteTrack configuration changed, reload it')
    values=tracker_config.validate(request.get('values'))
    if values==current:return
    active=run(['/usr/bin/systemctl','show',UNIT,'-p','ActiveState','--value']).strip()=='active'
    tracker_config.save(values)
    try:
        if not active:return
        run([*SERVICE_CMD,'restart',UNIT],45)
        expected=tracker_config.revision(values)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            state=status()
            if state['tracker_revision']==expected and len(state['fps'])==len(state['cameras']):return
            time.sleep(.5)
        raise RuntimeError('Edge workers did not confirm the ByteTrack revision')
    except Exception:
        tracker_config.save(current)
        if active:run([*SERVICE_CMD,'restart',UNIT],45)
        raise

def status():
    values=dict(x.split('=',1) for x in run(['/usr/bin/systemctl','show',UNIT,'-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','NRestarts','-p','MemoryCurrent','-p','ActiveEnterTimestampMonotonic','-p','InvocationID']).splitlines() if '=' in x)
    memory={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable']}
    fps,counts={},{}
    if values.get('ActiveState')=='active':
        log=run(['/usr/bin/journalctl','-u',UNIT,'--since','-5s','-n','80','-o','json','--no-pager'])
        fps,counts=camera_metrics(log,values.get('InvocationID'),time.time())
    tracker_values=tracker_config.load()
    tracker_revision=None
    if values.get('ActiveState')=='active':
        invocation=values.get('InvocationID')
        cameras=json.loads((BASE/'active_cameras.json').read_text())
        confirmations=[]
        for cam in cameras:
            try:confirmations.append((BASE/f'.tracker-worker-{int(cam)}').read_text())
            except FileNotFoundError:break
        expected=invocation+':'+tracker_config.revision(tracker_values)
        if confirmations==[expected]*len(cameras):tracker_revision=tracker_config.revision(tracker_values)
    disk=shutil.disk_usage('/')
    clock=None
    try:
        text=run(['/usr/bin/chronyc','-n','tracking'],2);m=re.search(r'System time\s*:\s*([\d.]+) seconds',text)
        if m:clock=float(m[1])*1000
    except Exception:pass
    models=[{'id':key,'file':Path(cfg['pt']).name} for key,cfg in model_catalog().items()]
    return {'unit_age_seconds':max(0,time.monotonic()-float(values.get('ActiveEnterTimestampMonotonic',0))/1e6),'unit':values,'fps':fps,'counts':counts,'cpu_percent':cpu_percent(),'gpu_percent':gpu_percent(),'gpu_memory_shared':True,'disk_used':disk.used,'disk_free':disk.free,'disk_total':disk.total,'memory_used':memory['MemTotal']-memory['MemAvailable'],'memory_total':memory['MemTotal'],'clock_offset_ms':clock,'model':json.loads(SELECTION.read_text()).get('name'),'models':models,'tracker_values':tracker_values,'tracker_config_revision':tracker_config.revision(tracker_values),'tracker_revision':tracker_revision,'cameras':json.loads((BASE/'active_cameras.json').read_text()),'time':time.time()}

try:
    action=os.environ.get('SSH_ORIGINAL_COMMAND','').strip()
    model_action=re.fullmatch(r'activate-model ('+MODEL_ID+r')',action)
    tracker_action=re.fullmatch(r'apply-tracker ([0-9a-f]{1,2048})',action)
    if not tracker_action and not model_action and action not in ('status','start','stop','restart'):raise ValueError('Unsupported action')
    if action!='status':
        lock=os.open(BASE,os.O_RDONLY|os.O_DIRECTORY)
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if tracker_action:apply_tracker(tracker_action[1])
            elif model_action:activate_model(model_action[1])
            else:run([*SERVICE_CMD,action,UNIT],45)
        finally:os.close(lock)
    print(json.dumps({'ok':True,'data':status()},separators=(',',':')))
except Exception as exc:
    print(json.dumps({'ok':False,'error':str(exc)[:240]}));raise SystemExit(1)
