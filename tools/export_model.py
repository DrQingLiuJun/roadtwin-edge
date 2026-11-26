#!/usr/bin/env python3
"""Run on a training/export machine; target Jetsons only need the resulting ONNX."""
import argparse,json,shutil
from pathlib import Path

def validate_shape(shape, height, width):
    if tuple(shape)!=(1,3,height,width):raise ValueError('Exported model has unexpected NCHW shape: '+str(shape))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('pt',type=Path);parser.add_argument('--width',type=int,required=True);parser.add_argument('--height',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if min(args.height,args.width)<=0 or args.height%32 or args.width%32:parser.error('Height and width must be positive multiples of 32')
    if args.output.suffix!='.onnx' or args.output.exists():parser.error('Use a new .onnx output path')
    from ultralytics import YOLO
    import onnx
    model=YOLO(str(args.pt));names={str(key):value for key,value in model.names.items()}
    if names!={'0':'person','1':'car','2':'cyclist','3':'truck'}:raise ValueError('Expected person/car/cyclist/truck class order')
    output=Path(model.export(format='onnx',imgsz=[args.height,args.width],batch=1,opset=12,dynamic=False,end2end=False,nms=False,simplify=False,half=False,device='cpu'))
    graph=onnx.load(str(output));onnx.checker.check_model(graph)
    inputs=[[d.dim_value for d in row.type.tensor_type.shape.dim] for row in graph.graph.input];outputs=[[d.dim_value for d in row.type.tensor_type.shape.dim] for row in graph.graph.output]
    if len(inputs)!=1 or len(outputs)!=1:raise ValueError('Expected one input and one output')
    validate_shape(inputs[0],args.height,args.width)
    if len(outputs[0])!=3 or outputs[0][0]!=1 or 8 not in outputs[0][1:]:raise ValueError('Expected dense four-class output; check end2end=False')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if output.resolve()!=args.output.resolve():shutil.move(str(output),str(args.output))
    print(json.dumps({'pt':str(args.pt),'onnx':str(args.output),'inputs':inputs,'outputs':outputs,'model_classes':names,'training_imgsz':model.ckpt.get('train_args',{}).get('imgsz'),'deployment_input_wh':[args.width,args.height]},ensure_ascii=False))

if __name__=='__main__':main()
