import argparse
import json
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
from PIL import Image
import yaml  # for torch hub
from util.base_cam import GradCAM,GradCAMPlusPlus,XGradCAM,LayerCAM#改写pytorch_grad_cam库，具有可视化特定通道的功能
from util.base_cam import show_cam_on_image
from util.component import YoloOutputTarget,DeepDream
from util.algorithm import ActivationsAndGradients
from tqdm import tqdm


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    new_args = parser.parse_args([])

    # global_config
    parser.add_argument('--config', action='store', type=str, default=None)

    # 
    parser.add_argument('--model',action='store', type=str, default='yolov5',help='model')


    # path_config
    parser.add_argument('--mpath',action='store', type=str, default='githubs/yolov5',help='relative path of model')
    parser.add_argument('--wpath',action='store', type=str, default='githubs/yolov5/weights/yolov5s.pt',help='relative path of weight')
    parser.add_argument('--ifolder',action='store', type=str, default='githubs/yolov5/data/images',help='relative folder path of image needs being processed')
    parser.add_argument('--ifile',action='store', type=str, default='dog',help='file name(wo .jpg etc) of image needs being processed,typically the class name, also the save folder of visualized results')
    parser.add_argument('--vpath','--spath',action='store', type=str, default='outputs/objectDetection/yolov5/layercam',help='relative save path of visualized image')
    parser.add_argument('--vmethod','--spath',action='store', type=str, default='layercam',help='visualized method')
    
    # deepdream_config
    parser.add_argument('--pyramid_size',action='store', type=int, default=3,help='pyramid_size in deepdream')
    parser.add_argument('--pyramid_ratio',action='store', type=int, default=2,help='deepdream configs')
    parser.add_argument('--gradient_ascent_epochs',action='store', type=int, default=10,help='deepdream configs')
    parser.add_argument('--deepdream_lr',action='store', type=float, default=1.5e-2,help='deepdream configs')
    parser.add_argument('--smoothing_coefficient',action='store', type=float, default=0.5,help='deepdream configs')
    parser.add_argument('--upper_img_bound',action='store', type=float, default=0.9,help='deepdream configs')
    parser.add_argument('--lower_img_bound',action='store', type=float, default=0.1,help='deepdream configs')


    parser.add_argument('--show_box',action='store_true',help='is show box')

    parser.add_argument('--is_channel',action='store_true',help='is visualize per channel')
    parser.add_argument('--channel',action='store', type=int, default=0,help='')




    # parser.add_argument('--plm_name', action='store', type=str, default='t5-base')
    # parser.add_argument('--plm_cache_path', action='store', type=str, default='./PLM_cache')
    # parser.add_argument('--mode', action='store', type=str)
    # parser.add_argument('--default_tokenizer', action='store_true')
    # parser.add_argument('--vocab_path', action='store', type=str)###
    # parser.add_argument('--kg_path', action='store', type=str)###
    # parser.add_argument('--ent_id_path', action='store')###
    # parser.add_argument('--rel_id_path', action='store')###
    # parser.add_argument('--database_path', action='store')###
    # parser.add_argument('--ddp_on', action='store_true')

    # #train_config
    # parser.add_argument('--epoch', action='store', type=int)
    # parser.add_argument('--batch_size', action='store', type=int)
    # parser.add_argument('--eval_epoch_interval', action='store', type=int)
    # parser.add_argument('--dataset', action='store', type=str)
    # parser.add_argument('--data_path', action='store', type=str)
    # parser.add_argument('--train_name', action='store', type=str)
    # parser.add_argument('--train_ckpt', action='store', type=str)
    # parser.add_argument('--output_path', action='store', type=str, default='./outputs')
    # parser.add_argument('--add_ent', action='store', type=bool)
    # ##optimizer
    # parser.add_argument('--train_config', action='store', type=str, default='configs/normal.json')
    # parser.add_argument('--lr', action='store', type=float)
    # parser.add_argument('--op_gamma', action='store', type=float)
    # parser.add_argument('--op_step', action='store', type=int)
    
    # ##test_config
    # parser.add_argument('--test_ckpt', action='store', type=str)
    # parser.add_argument('--infer_json', action='store', type=str)###

    # ##rl
    # parser.add_argument('--rl_epoch', action='store', type=int)
    # parser.add_argument('--rl_batch_size', action='store', type=int)
    # parser.add_argument('--rl_lr', action='store', type=float)
    # parser.add_argument('--episode_periter', action='store')
    # parser.add_argument('--eval_pervaliditer', action='store', type=int)
    # parser.add_argument('--rlbase_ckpt', action='store', type=str)
    # parser.add_argument('--rljson_path', action='store', type=str)



    # #ablation
    # parser.add_argument('--vocabulary_adjusting', action='store_true')


    args = parser.parse_args()

    if args.config is not None:
        with open(args.config,'r') as f:
            configs = json.load(f)
        for k,v in configs.items():
            setattr(new_args,k,v)
        for attr, value in vars(args).items():
            if value is not None:
                setattr(new_args,attr,value)
    else:
        new_args=args

    return new_args


def model_init(args):
    yolov5_hub=os.path.join(os.getcwd(),args.mpath)
    weights_path=os.path.join(os.getcwd(),args.wpath)
    # print(yolov5_hub,weights_path)
    # assert False
    model = torch.hub.load(yolov5_hub, 'custom', path=weights_path, source='local')

    

    return model

def img_init(args):
    img_name=args.ifile#bus,cat,cloud,dinner,dog,farm,food,puppies,robot,zidane
    img_path = os.path.join(os.getcwd(),args.ifolder,'{}.jpg'.format(img_name)) # or file, Path, PIL, OpenCV, numpy, list
    save_folder_path=os.path.join(os.getcwd(),args.vpath,args.ifile)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    save_img_path=os.path.join(save_folder_path,f'{args.ifile}.jpg')
    save_img_path='/home/mdisk1/zhongchulong/interpt/outputs/objectDetection/yolov5/test/layercam/cow/cow.jpg'

    
    
    
    if 'zero' in img_name:
        img=torch.zeros((1,3,640,640),requires_grad=True)
        rgb_img=img.clone().squeeze(0).permute(1,2,0).detach().numpy()*255
        # cv.imwrite(save_img_path,rgb_img)
        img=img.cuda()
        img.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度
        
    else:
        img = cv.imread(img_path)
        # img = cv.imread(save_img_path)
        img = cv.resize(img, (640, 640))
        # img = cv.resize(img, (1280, 1280))#deepdream必须要这个
        # cv.imwrite(save_img_path,img)
        img=cv.cvtColor(img, cv.COLOR_BGR2RGB) #deepdream必须要这个
        img = np.float32(img) / 255
        rgb_img = img.copy()#用于cam和原图的结合
        img = transforms.ToTensor()(img).unsqueeze(0)#(b,c,h,w)
        pil_img=transforms.ToPILImage()(img.squeeze(0))
        pil_img.save(save_img_path)
        img=img.cuda()#用于输入visual_cal进行可视化
        # img.requires_grad=True
        # img.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度

    

    return rgb_img,img


def draw_box(img, box, color, name):
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
    cv.putText(img, str(name), (xmin, ymin - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv.LINE_AA)
    return img

def visual_cal(args,img,model,layer):
    img_name=args.ifile#bus,cat,cloud,dinner,dog,farm,food,puppies,robot,zidane
    img_path = os.path.join(os.getcwd(),args.ifolder,'{}.jpg'.format(img_name))
    # img_path=os.path.join(os.getcwd(),args.vpath,img_name,f'{img_name}_ori.jpg')

    if 'yolo' in args.model.lower():
        model_autoshape=model
        results=model_autoshape(img_path)
        # results=model_autoshape.model.model(img)
        # results=model_autoshape(img)

        # class_dict=results.names
        # pred=results.xywhn[0]
        # print(pred)

        
        model=model_autoshape.model
        model.requires_grad_(True)#本地模式需要手动启动梯度回传
        target_layers = [model.model.model[layer]]


        # assert False

        targets = [YoloOutputTarget(results,is_max_cls=True)]
    else:
        print('args.model error')
        assert False

    # targets=None
        

    if 'cam' in args.vmethod.lower():
        if args.vmethod.lower()=='layercam':
            cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=False,is_channel=args.is_channel,channel=args.channel)
        elif args.vmethod.lower()=='gradcam':
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False,is_channel=args.is_channel,channel=args.channel)
        elif args.vmethod.lower()=='gradcam++':
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False,is_channel=args.is_channel,channel=args.channel)
        elif args.vmethod.lower()=='xgradcam':
            cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=False,is_channel=args.is_channel,channel=args.channel)
        else:
            print('args.vmethod error, we use the default layercam')
            cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=False,is_channel=args.is_channel,channel=args.channel)
        grayscale_cam = cam(input_tensor=img, targets=targets)
        visual_results = grayscale_cam[0, :,:]
    elif 'forward' in args.vmethod.lower():
        aag = ActivationsAndGradients(model=model, target_layers=target_layers, is_channel=args.is_channel,channel=args.channel)
        img_outputs = aag(img)
        feature_map=aag.activations[0]
        feature_map=nn.UpsamplingBilinear2d(size=(640,640))(feature_map)##([1, 32, 320, 320])->[1, 32, 640, 640])
        visual_results=torch.mean(feature_map[0], dim=0)#([1, 32, 640, 640])->[640, 640]
        visual_results=visual_results.cpu().detach()

    elif 'deepdream' in args.vmethod.lower():
        dd = DeepDream(model=model, target_layers=target_layers, args=args)
        feature_map = dd(img)
        

        # feature_map=nn.UpsamplingBilinear2d(size=(640,640))(feature_map)##([1, 3, 320, 320])->[1, 3, 640, 640])


        #无to_pil
        visual_results=torch.mean(feature_map, dim=0)#([1, 3, 640, 640])->[3, 640, 640]
        visual_results=visual_results.permute(1,2,0)*255#([3, 640, 640])->[640, 640, 3]
        visual_results=visual_results.cpu().detach()

        # visual_results=feature_map#有to_pil

        del dd

    else:
        print('args.vmethod error')
        assert False


    return visual_results,results

def visual_save(args,img,results,grayscale_cam,layer):
    # print('grayscale_cam.shape={}'.format(grayscale_cam.shape))

    color=[1.,0.,0.]
    class_dict=results.names#0~79
    preds=results.xyxy[0]#[4,6]
    print('---preds---')
    print(preds)
    if args.show_box:
        for pred in preds:
            pred = pred.cpu().detach().numpy()
            img = draw_box(img,pred[:4], color, f'{class_dict[int(pred[5])]} {float(pred[4]):.2f}')
    if 'deepdream' in args.vmethod.lower():
        visualization=grayscale_cam#deepdream可视化直接保存输出，不需要mask和原图重叠
        # #有to_pil
        # to_pil = transforms.ToPILImage()
        # visualization = to_pil(visualization.squeeze(axis=0))
    else:
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    visualization = Image.fromarray(np.uint8(visualization))


    # channel=4
    # visualization.save(os.path.join(os.getcwd(),visual_path,'yolov5_layercam_layer_{}_ch_{}.jpg'.format(layer,channel)))

    if args.is_channel:
        save_img_name='{}_{}_layer_{}_ch_{}.jpg'.format(args.model.lower(),args.vmethod.lower(),layer,args.channel)
    else:
        save_img_name='{}_{}_layer_{}.jpg'.format(args.model.lower(),args.vmethod.lower(),layer)
    visualization.save(os.path.join(os.getcwd(),args.vpath,args.ifile,save_img_name))
    


def main(args):
    model=model_init(args)#模型初始化
    rgb_img,img=img_init(args)#输入图片预处理
    print(img.shape)
    if 'yolo' in args.model.lower():
        layer_num=len(model.model.model.model)-1#yolo最后一层是proposal，无法可视化
    else:
        print('args.model error')
        assert False

    # for layer in range(layer_num):
    for layer in range(0,layer_num):
        grayscale_cam,results=visual_cal(args,img,model,layer)#可视化运算
        visual_save(args,rgb_img,results,grayscale_cam,layer)#可视化结果保存
        del grayscale_cam

if __name__=='__main__':
    args = parse_args()#获取命令行参数
    print(args)
    main(args)

    # with open('config/yolo_forward.json','w') as f:
    #     json.dump(vars(args),f,indent=4)