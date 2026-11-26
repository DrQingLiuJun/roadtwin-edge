#!/usr/bin/env python3
"""Manage local model manifests without placing device binaries in Git."""
import argparse,hashlib,json,os,re,subprocess,sys,tempfile
from datetime import date
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
CLASSES={'0':'person','1':'car','2':'cyclist','3':'truck'}
WIRE={'0':0,'1':2,'2':1,'3':7}
MODEL_ID=r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}'

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()

def local_file(name, parent='models'):
    path=(BASE/name).resolve()
    if not path.is_relative_to((BASE/parent).resolve()) or not path.is_file():raise ValueError('Expected local '+parent+' file: '+str(name))
    return path

def catalog():
    result={}
    for manifest in sorted((BASE/'models').glob('*/selection.json')):
        if manifest.is_symlink() or manifest.parent.is_symlink():continue
        data=json.loads(manifest.read_text())
        if not re.fullmatch(MODEL_ID,data.get('name','')):raise ValueError('Invalid model ID in '+str(manifest))
        if data['name'] in result:raise ValueError('Duplicate model ID: '+data['name'])
        result[data['name']]=data
    return result

def expected_shape(config):
    size=config['input_size'];height,width=(size,size) if type(size) is int else size
    if type(height) is not int or type(width) is not int or min(height,width)<=0 or height%32 or width%32:raise ValueError('Input height and width must be positive multiples of 32')
    return (1,3,height,width)

def engine_bindings(path):
    import tensorrt as trt
    logger=trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine=runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:raise ValueError('Unable to load TensorRT engine')
        return [{'name':engine.get_binding_name(i),'shape':tuple(engine.get_binding_shape(i)),'input':engine.binding_is_input(i),'dtype':str(engine.get_binding_dtype(i))} for i in range(engine.num_bindings)]

def verify(config, inference=True):
    if config.get('model_classes')!=CLASSES or config.get('wire_class_map')!=WIRE:raise ValueError('The deployed class order and wire IDs must be preserved')
    if config.get('library')!='runtime/libdesktop_detector.so':raise ValueError('Unexpected inference library')
    for field in ('pt','engine'):
        path=local_file(config[field])
        if digest(path)!=config[field+'_sha256']:raise ValueError(field+' SHA-256 mismatch')
    bindings=engine_bindings(local_file(config['engine']));inputs=[row for row in bindings if row['input']];outputs=[row for row in bindings if not row['input']]
    if len(inputs)!=1 or inputs[0]['shape']!=expected_shape(config):raise ValueError('TensorRT input differs from selection manifest')
    if len(outputs)!=1 or len(outputs[0]['shape'])!=3 or outputs[0]['shape'][0]!=1 or 8 not in outputs[0]['shape'][1:]:raise ValueError('Expected dense four-class output [1,8,N] or [1,N,8], not end-to-end [1,300,6]')
    if inference:
        import numpy as np
        sys.path.insert(0,str(BASE/'runtime'))
        from desktop_engine import DesktopDetector
        os.environ['RSU_MODEL_NUM_CLASSES']='4';os.environ['RSU_MODEL_CLASS_MAP']='0,1,2,3'
        detector=DesktopDetector(local_file(config['engine']),BASE/config['library'])
        try:detector.predict(np.zeros((expected_shape(config)[2],expected_shape(config)[3],3),dtype=np.uint8))
        finally:detector.close()
    return {'name':config['name'],'verified':True,'bindings':bindings,'inference_smoke_test':inference}

def append_log(message):
    readme=BASE/'README.md';text=readme.read_text();heading='## 更新日志\n';position=text.find(heading)
    if position<0:raise ValueError('README 更新日志 section is missing')
    position+=len(heading);entry='\n- '+date.today().isoformat()+'：'+message+'\n'
    atomic_write(readme,(text[:position]+entry+text[position:]).encode())

def atomic_write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.model-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as file:file.write(data);file.flush();os.fsync(file.fileno())
        os.chmod(name,0o644);os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def activate(name):
    if name not in catalog():raise ValueError('Unknown model: '+name)
    verify(catalog()[name])
    env=dict(os.environ,SSH_ORIGINAL_COMMAND='activate-model '+name)
    result=subprocess.run(['/usr/bin/python3','-B',str(BASE/'ops_agent.py')],env=env,capture_output=True,text=True,timeout=100)
    if result.returncode:raise RuntimeError(result.stdout[-500:]+result.stderr[-200:])
    response=json.loads(result.stdout)
    if not response.get('ok'):raise RuntimeError(response.get('error','Activation failed'))
    print(json.dumps({'activated':response['data']['model'],'fps':response['data']['fps']},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('list')
    check=commands.add_parser('verify');check.add_argument('name',nargs='?');check.add_argument('--all',action='store_true');check.add_argument('--bindings-only',action='store_true')
    switch=commands.add_parser('activate');switch.add_argument('name')
    add=commands.add_parser('register');add.add_argument('--name',required=True);add.add_argument('--pt',required=True);add.add_argument('--engine',required=True);add.add_argument('--width',type=int,required=True);add.add_argument('--height',type=int,required=True);add.add_argument('--precision',choices=['FP16','FP32'],required=True);add.add_argument('--training-imgsz',type=int,required=True);add.add_argument('--replace',action='store_true')
    args=parser.parse_args()
    if args.command=='list':
        for name,config in catalog().items():print(json.dumps({'name':name,'input_nchw':expected_shape(config),'precision':config.get('precision'),'engine':config['engine']},ensure_ascii=False))
    elif args.command=='verify':
        models=catalog();names=list(models) if args.all else [args.name or json.loads((BASE/'selected_model.json').read_text())['name']]
        for name in names:print(json.dumps(verify(models[name],not args.bindings_only)))
    elif args.command=='activate':activate(args.name)
    else:
        if not re.fullmatch(MODEL_ID,args.name):raise ValueError('Invalid model ID')
        if args.name in catalog() and not args.replace:raise ValueError('Model ID already exists; use another ID or explicit --replace')
        folder=local_file(args.pt).parent;engine=local_file(args.engine)
        if engine.parent!=folder:raise ValueError('Place PT and engine in the same model directory')
        manifest=folder/'selection.json'
        if manifest.exists() and not args.replace:raise ValueError('Model directory is already registered')
        config={'name':args.name,'pt':str(local_file(args.pt).relative_to(BASE)),'engine':str(engine.relative_to(BASE)),'library':'runtime/libdesktop_detector.so','pt_sha256':digest(local_file(args.pt)),'engine_sha256':digest(engine),'input_size':[args.height,args.width],'input_width':args.width,'input_height':args.height,'num_classes':4,'precision':args.precision,'training_imgsz':args.training_imgsz,'model_classes':CLASSES,'wire_class_map':WIRE}
        verify(config);previous=manifest.read_bytes() if manifest.exists() else None
        try:
            atomic_write(manifest,(json.dumps(config,ensure_ascii=False,indent=2)+'\n').encode())
            append_log('注册模型 '+args.name+'，输入 '+str(args.width)+'×'+str(args.height)+'，'+args.precision+'；完成权重、引擎哈希与本机推理校验。')
        except Exception:
            if previous is None:manifest.unlink(missing_ok=True)
            else:atomic_write(manifest,previous)
            raise
        print(json.dumps({'registered':args.name,'manifest':str(manifest),'activation':'Use the console or tools/models.py activate'},ensure_ascii=False))

if __name__=='__main__':main()
