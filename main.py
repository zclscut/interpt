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
    parser.add_argument('--mpath','-mp',action='store', type=str, default='githubs/yolov5',help='relative Path of Model')
    parser.add_argument('--wpath','-wp',action='store', type=str, default='githubs/yolov5/weights/yolov5s.pt',help='relative Path of Weight')
    parser.add_argument('--ifolder','-ifo',action='store', type=str, default='githubs/yolov5/data/images',help='relative Folder path of Image needs being processed')
    parser.add_argument('--ifile','-ifi',action='store', type=str, default='dog',help='File name(wo .jpg etc) of Image needs being processed,typically the class name, also the save folder of visualized results')
    parser.add_argument('--vpath','--spath','-vp','-sp',action='store', type=str, default='outputs/objectDetection/yolov5/layercam',help='relative Save Path of Visualized image')
    parser.add_argument('--vmethod','-vm',action='store', type=str, default='layercam',help='Visualized Method')
    
    # deepdream_config
    parser.add_argument('--pyramid_size',action='store', type=int, default=3,help='pyramid_size in deepdream')
    parser.add_argument('--pyramid_ratio',action='store', type=int, default=2,help='deepdream configs')
    parser.add_argument('--gradient_ascent_epochs',action='store', type=int, default=10,help='deepdream configs')
    parser.add_argument('--deepdream_lr',action='store', type=float, default=1.5e-2,help='deepdream configs')
    parser.add_argument('--smoothing_coefficient',action='store', type=float, default=0.5,help='deepdream configs')
    parser.add_argument('--upper_img_bound',action='store', type=float, default=0.9,help='deepdream configs')
    parser.add_argument('--lower_img_bound',action='store', type=float, default=0.1,help='deepdream configs')


    parser.add_argument('--show_box',action='store_true',help='is show box')
    parser.add_argument('--inverse_color',action='store_true',help='color of box, always red when false, changing with background when true')


    parser.add_argument('--is_channel',action='store_true',help='is visualize per channel')
    parser.add_argument('--channel',action='store', type=int, default=0,help='enable when is_channel is true')



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
    # save_img_path=os.path.join(save_folder_path,f'{args.ifile}.jpg')
    

    
    
    
    if 'zero' in img_name:
        img=torch.zeros((1,3,640,640),requires_grad=True)
        img_ori=img.clone()#为了和else统一方便return，没有其他意义
        rgb_img=img_ori.squeeze(0).permute(1,2,0).detach().numpy()*255
        cv.imwrite(img_path,rgb_img)
        img=img.cuda()
        img.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度
        
    else:
        img_ori = cv.imread(img_path)
        # img = cv.imread(save_img_path)

        img = cv.resize(img_ori, (640, 640))
        # img = cv.resize(img, (1280, 1280))#deepdream必须要这个
        img=cv.cvtColor(img, cv.COLOR_BGR2RGB) #deepdream必须要这个
        img = np.float32(img) / 255
        rgb_img = img.copy()#用于cam和原图的结合
        img = transforms.ToTensor()(img).unsqueeze(0)#(b,c,h,w)

        # pil_img=transforms.ToPILImage()(img.clone().squeeze(0))
        # pil_img.save(save_img_path)
        img=img.cuda()#用于输入visual_cal进行可视化
        # img.requires_grad=True
        # img.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度

    

    return rgb_img,img,img_ori


def draw_box(args,img, box, name):
    xmin, ymin, xmax, ymax = list(map(int, list(box)))

    #根据背景颜色自适应调整框的颜色
    if args.inverse_color:
        bg_color=img[xmin:xmax,ymin:ymax,...].mean(axis=0).mean(axis=0)#背景平均色
        inverse_color=(np.array([255.,255.,255.])-bg_color)#反色
        color=inverse_color
        # color=tuple(x for x in color)#三原色比例混合
        # color=tuple(int(x) for x in color)#基础颜色非0即1，取三原色的组合，非比例混合。
    else:
        color=(255,0,0)#固定的红色


    cv.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv.putText(img, str(name), (xmin, ymin - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv.LINE_AA)
    return img

def visual_cal(args,img,model,layer):
    img_name=args.ifile#bus,cat,cloud,dinner,dog,farm,food,puppies,robot,zidane
    img_path = os.path.join(os.getcwd(),args.ifolder,'{}.jpg'.format(img_name))#1280×640
    # img_path=os.path.join(os.getcwd(),args.vpath,img_name,f'{img_name}.jpg')#640×640

    if 'yolo' in args.model.lower():
        model_autoshape=model
        results=model_autoshape(img_path)
        
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

def visual_save(args,img,img_ori,results,grayscale_cam,layer):
    # print('grayscale_cam.shape={}'.format(grayscale_cam.shape))

    
    if 'deepdream' in args.vmethod.lower():
        visualization=grayscale_cam#deepdream可视化直接保存输出，不需要mask和原图重叠
        # #有to_pil
        # to_pil = transforms.ToPILImage()
        # visualization = to_pil(visualization.squeeze(axis=0))
    else:
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)


        scale_x_ratio=img_ori.shape[1]/640.
        scale_y_ratio=img_ori.shape[0]/640.

        # color=[1.,0.,0.]
        class_dict=results.names#0~79
        preds=results.xyxy[0]#[4,6]
        print('---preds---')
        print(preds)
        if args.show_box:
            for pred in preds:
                pred = pred.cpu().detach().numpy()
                pred[0]=pred[0]/scale_x_ratio
                pred[2]=pred[2]/scale_x_ratio
                pred[1]=pred[1]/scale_y_ratio
                pred[3]=pred[3]/scale_y_ratio

                # img = draw_box(img,pred[:4], color, f'{class_dict[int(pred[5])]} {float(pred[4]):.2f}')
                visualization = draw_box(args,visualization,pred[:4], f'{class_dict[int(pred[5])]} {float(pred[4]):.2f}')
    
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
    rgb_img,img,img_ori=img_init(args)#输入图片预处理
    print(img.shape)
    if 'yolo' in args.model.lower():
        layer_num=len(model.model.model.model)-1#yolo最后一层是proposal，无法可视化
    else:
        print('args.model error')
        assert False

    # for layer in range(layer_num):
    for layer in range(0,layer_num):
        grayscale_cam,results=visual_cal(args,img,model,layer)#可视化运算
        visual_save(args,rgb_img,img_ori,results,grayscale_cam,layer)#可视化结果保存
        del grayscale_cam

if __name__=='__main__':
    args = parse_args()#获取命令行参数
    print(args)
    main(args)

    # with open('config/yolo_forward.json','w') as f:
    #     json.dump(vars(args),f,indent=4)