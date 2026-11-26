#!/usr/bin/env python3
"""One fresh resource snapshot per second; no inference or control code runs here."""
import json,logging,os,signal,subprocess,threading,time,uuid
from pathlib import Path
from redis_io import RedisIO

BASE=Path(__file__).resolve().parent
STOP=threading.Event()

def collect(agent):
    env=dict(os.environ,SSH_ORIGINAL_COMMAND='status')
    result=subprocess.run(['/usr/bin/python3','-B',agent],env=env,capture_output=True,text=True,timeout=6)
    if result.returncode:raise RuntimeError('Resource sampler failed: '+(result.stdout or result.stderr)[-240:])
    response=json.loads(result.stdout)
    if response.get('ok') is not True or not isinstance(response.get('data'),dict):raise ValueError('Invalid sampler response')
    data=response['data'];stamp=data.get('time')
    if type(stamp) not in (int,float) or not 0<=time.time()-stamp<3:raise ValueError('Resource sample is stale')
    return data

def envelope(config, metrics, session, sequence, started):
    sampled=int(metrics['time']*1000)
    return {'schema_version':1,'payload_schema':'roadtwin.edge_resources.v1','source':{'id':config['id'],'ip':config['ip'],'hardware':'Jetson AGX Orin','cameras':config['cameras'],'camera_kind':config.get('camera_kind','rsu')},'time':{'sampled_at_ms':sampled,'generated_at_ms':round(time.time()*1000),'session':session,'sequence':sequence,'target_upload_hz':1,'collection_ms':round((time.monotonic()-started)*1000,1)},'metrics':metrics}

def write_health(path, value):
    try:
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n');temporary.replace(path)
    except OSError:logging.exception('Unable to write telemetry health')

def main():
    config=json.loads((BASE/'settings.json').read_text());client=RedisIO(config['redis_host'],config['redis_port'])
    session=uuid.uuid4().hex;sequence=0;successful=0;failed=0;last_success=None;last_log=0
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    for event in (signal.SIGTERM,signal.SIGINT):signal.signal(event,lambda *_:STOP.set())
    logging.info('Publishing %s to %s at 1 Hz',config['id'],config['key'])
    while not STOP.is_set():
        started=time.monotonic();sequence+=1;error=None;size=None
        try:
            metrics=collect(config['ops_agent']);payload=envelope(config,metrics,session,sequence,started)
            encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            if len(encoded.encode())>64*1024:raise ValueError('Resource payload exceeds limit')
            client.set(config['key'],encoded,15);last_success=time.time();successful+=1;size=len(encoded.encode())
        except Exception as exc:
            client.close();failed+=1;error=str(exc)[:240]
            if time.monotonic()-last_log>=15:logging.warning('Telemetry upload failed: %s',error);last_log=time.monotonic()
        write_health(BASE/'status.json',{'id':config['id'],'key':config['key'],'updated':time.time(),'last_success':last_success,'sequence':sequence,'successful':successful,'failed':failed,'error':error,'payload_bytes':size,'cycle_ms':round((time.monotonic()-started)*1000,1)})
        STOP.wait(max(0,1-(time.monotonic()-started)))
    client.close()

if __name__=='__main__':main()
