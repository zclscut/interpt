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
from util.component import YoloOutputTarget
from util.algorithm import ActivationsAndGradients

from torch.autograd import Variable


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


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
    parser.add_argument('--apath','--spath',action='store', type=str, default='githubs/yolov5/data/images/fgsm',help='relative save path of adversarial image')
    parser.add_argument('--amethod','--smethod',action='store', type=str, default='fgsm',help='adversarial method')
    


    parser.add_argument('--iter',action='store', type=int, default=2,help='ifgsm only')
    parser.add_argument('--alpha',action='store', type=int, default=1,help='ifgsm only')
    parser.add_argument('--eps',action='store', type=float, default=0.03,help='ifgsm and fgsm')
    parser.add_argument('--x_val_min',action='store', type=float, default=0.,help='ifgsm and fgsm')
    parser.add_argument('--x_val_max',action='store', type=float, default=1.,help='ifgsm and fgsm')



    # parser.add_argument('--is_channel',action='store_true',help='is visualize per channel')
    # parser.add_argument('--channel',action='store', type=int, default=0,help='')


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

    #创建保存对抗样本文件夹
    adv_save_folder=os.path.join(os.getcwd(),args.apath)
    print(adv_save_folder)
    if not os.path.exists(adv_save_folder):
        os.makedirs(adv_save_folder)
    
    
    img = cv.imread(img_path)
    img = cv.resize(img, (640, 640))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    img = transforms.ToTensor()(img).unsqueeze(0)#(b,c,h,w)
    img=img.cuda()
    # img.requires_grad=True
    # img.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度

    

    return rgb_img,img


def adver(args,model,img):
    img_name=args.ifile#bus,cat,cloud,dinner,dog,farm,food,puppies,robot,zidane
    img_path = os.path.join(os.getcwd(),args.ifolder,'{}.jpg'.format(img_name))

    model_autoshape=model
    model_autoshape.requires_grad_(True)
    results = model_autoshape(img_path)


    model=model_autoshape.model


    model.requires_grad_(True)#本地模式需要手动启动梯度回传
    target_layers = [model.model.model[0]]
    targets = [YoloOutputTarget(results,is_max_cls=True)]
    aag = ActivationsAndGradients(model=model, target_layers=target_layers)

    if args.amethod=='fgsm':
        
        # x_adv=img.clone()
        x_adv=img
        x_adv.requires_grad_(True)
        x_outputs = aag(x_adv)

        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss_ls=[target(output) for target, output in zip(targets, x_outputs)]
        loss = sum(loss_ls)
        # loss.requires_grad=True
        loss.backward(retain_graph=True)


        # eps,x_val_min,x_val_max=0.03,0,1
        eps,x_val_min,x_val_max=args.eps,args.x_val_min,args.x_val_max
        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        x_adv=x_adv.squeeze(0)
        x_adv = np.uint8(x_adv.detach().cpu().numpy()* 255)
        x_adv=np.transpose(x_adv, (1, 2, 0))
        adv_save_path=os.path.join(os.getcwd(),args.apath,'{}.jpg'.format(img_name))
    elif args.amethod=='ifgsm':
        x_adv=img
        x_adv.requires_grad_(True)
        # x_adv = Variable(x.data, requires_grad=True)

        iteration=args.iter#2,3,5
        alpha,eps,x_val_min,x_val_max=args.alpha,args.eps,args.x_val_min,args.x_val_max
        for i in range(iteration):
            # x_adv.requires_grad_(True)

            x_outputs = aag(x_adv)


            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss_ls=[target(output) for target, output in zip(targets, x_outputs)]
            loss = sum(loss_ls)
            # loss.requires_grad=True
            
            if not hasattr(loss, 'requires_grad'):
                break


            loss.backward(retain_graph=True)
            # activations_list = [a.cpu().data.numpy()
            #                             for a in aag.activations]
            # grads_list = [g.cpu().data.numpy()
            #                 for g in aag.gradients]
            # print(len(aag.activations))
            # print(len(aag.gradients))

            
            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > img+eps, img+eps, x_adv)
            x_adv = where(x_adv < img-eps, img-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        x_adv=x_adv.squeeze(0)
        x_adv = np.uint8(x_adv.detach().cpu().numpy()* 255)
        x_adv=np.transpose(x_adv, (1, 2, 0))
        adv_save_folder=os.path.join(os.getcwd(),args.apath,f'iter{iteration}')
        if not os.path.exists(adv_save_folder):
            os.makedirs(adv_save_folder)
        adv_save_path=os.path.join(os.getcwd(),args.apath,'iter{}/{}.jpg'.format(iteration,img_name))
    


    
    print(cv.imwrite(adv_save_path,x_adv))



def main(args):#处理一张图片
    model=model_init(args)#模型初始化
    rgb_img,img=img_init(args)#输入图片预处理
    adver(args,model,img)


def get_img_name(ifile_name):
    ifile=''
    if ifile_name.endswith('.jpg'):
        ifile=ifile_name.rstrip('.jpg')
    elif ifile_name.endswith('.png'):
        ifile=ifile_name.rstrip('.png')
    elif ifile_name.endswith('.jpeg'):
        ifile=ifile_name.rstrip('.jpeg')
    return ifile

if __name__=='__main__':
    args = parse_args()#获取命令行参数
    print(args)
    ifiles=os.listdir(os.path.join(os.getcwd(),args.ifolder))
    ifiles=[ifile_name for ifile_name in ifiles if ifile_name.endswith('.jpg') or ifile_name.endswith('.png') or ifile_name.endswith('.jpeg')]
    # print(ifiles)
    for ifile in ifiles:
        if ifile=='':
            break
        args.ifile=get_img_name(ifile)
        main(args)

    # with open('config/yolo_forward.json','w') as f:
    #     json.dump(vars(args),f,indent=4)