import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.nn import functional as F
import torchvision
import numpy as np
import os
from .algorithm import CascadeGaussianSmoothing,ActivationsAndGradients

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]


class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return torch.abs(model_output) * sign


class SoftmaxOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.softmax(model_output, dim=-1)


class RawScoresOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output


class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output



#nms逆向
class YoloOutputTarget:
    def __init__(self, results,is_max_cls=False):
        
        self.class_dict=results.names#0~79
        self.output_nms=results.xywh[0]#[4,6]
        # self.output_nms.requires_grad=True
        self.is_max_cls=is_max_cls

    def __call__(self, output_wo_nms):#[1, 25200, 85]
        class_dict=self.class_dict
        output_nms=self.output_nms.clone()
        output_nms.requires_grad=True


        if self.is_max_cls:# highest class conf only
            # print(output_nms)
            class_max_index=output_nms.argmax(dim=0)[4]
            output_nms=output_nms[class_max_index].unsqueeze(0)
            # print(output_nms)
            # assert False





        # output_wo_nms=
        output_wo_nms_reshape=None
        for i in range(output_wo_nms.shape[0]):#(batch,25200,85)->(25200,6)
            max_pred,max_index  = output_wo_nms[i, 5:].max(0, keepdim=True)#argmax
            class_label=max_index.float()
            reshape_tensor=torch.cat((output_wo_nms[i,0:5],class_label)).unsqueeze(0)
            if i==0:   
                output_wo_nms_reshape=reshape_tensor
            else:
                output_wo_nms_reshape=torch.cat((output_wo_nms_reshape,reshape_tensor),axis=0)


        output_index=[]
        conf_ls=[]
        conf_wo_nms_max=0
        for i in range(output_nms.shape[0]):
            for j in range(output_wo_nms_reshape.shape[0]):
                if output_nms[i,5]==output_wo_nms_reshape[j,5]:#cls
                    decimals=2
                    
                
                    if self.is_max_cls:
                        conf_nms=output_nms[i,4]
                        c=int(output_nms[i,5].item())
                        conf_wo_nms=output_wo_nms_reshape[j,4]*output_wo_nms[j,4+1+c]
                        
                        if output_index==[]:
                            output_index.append((c,j))
                            conf_wo_nms_max=conf_wo_nms
                        elif conf_wo_nms>conf_wo_nms_max:
                            conf_wo_nms_max=conf_wo_nms
                            output_index=[]
                            output_index.append((c,j))
                        # if conf_wo_nms.item()>0.1:
                        #     print(conf_nms.item(),conf_wo_nms.item())
                        #     print(output_wo_nms_reshape[j])
                        # if torch.round(conf_nms,decimals=decimals)==torch.round(conf_wo_nms,decimals=decimals):
                        #     conf_ls.append(conf_wo_nms)
                        #     print(c)

                    else:
                        if torch.round(output_nms[i,4],decimals=decimals)==torch.round(output_wo_nms_reshape[j,4],decimals=decimals):#conf,五位小数内相等
                            output_index.append((int(output_nms[i,5].item()),j))#(index,class)
            # assert False


        # print(output_index)
        conf_detect=[output_wo_nms[j,4] for (c,j) in output_index]#detect confidence
        conf_class=[output_wo_nms[j,4+1+c] for (c,j) in output_index]#classify confidence
        pred_class=[c for (c,j) in output_index]

        # print('---pred_class---')
        # for c in pred_class:
        #     print(c,self.class_dict[c])
        # print('---conf_detect---')
        # print(conf_detect)
        # print('---conf_class---')
        # print(conf_class)

        conf_ls=conf_detect+conf_class
        # conf_ls=conf_detect
        # conf_ls=conf_class
        
        # print('---conf_ls---')
        # print(conf_ls)

        conf_sum=sum(conf_ls)#使用.clone()方法，你创建副本是一个叶子节点



        
        return conf_sum
    

class ForwardMap(nn.Module):
    """
	
	"""
    def __init__(self,model,cfg):
        """初始化特征图存储类，每一次forward会存储每一层输出的特征图，网络层由cfg中的.yaml定义

		ForwardMap类会建立model的一个副本self.model，每次前向过程都会更新一次副本的特征图存储
		
		Args:
			model: 要存储输出特征图的原模型,需要是nn.Sequential
			cfg: 对应原模型模型权重文件.yaml的路径
		"""
        super().__init__()
        
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)  # model dict
        self.input={}
        for i, (f, n, m, args) in enumerate(self.yaml['backbone'] + self.yaml['head']):#读取输入域
            self.input[i]=f

        self.model=nn.Sequential()#拷贝的副本
        self.origin=model



    def forward(self, x):
        # print(self.origin)
        i=0
        self.model=nn.Sequential()#清空序列
        for name,module in self.origin.named_children():#m=('conv1', Conv2d(3, 6, 5))
            # print(i,name)
            # print('self.input[i]={}'.format(self.input[i]))
            if self.input[i] != -1:  # 根据yaml定义的输入域，如果是j=-1(上一层)或j=5，则返回对应层的特征图；如果j是列表，则也返回特征图列表，例如concat操作有多个输入域
                x = self.model[i].feature_map if isinstance(self.input[i], int) else [x if j == -1 else self.model[j].feature_map for j in self.input[i]]
            i=i+1
            # if len(x)==2:
            #     print(x[0].shape,x[1].shape)
            # elif len(x)==3:
            #     print(x[0].shape,x[1].shape,x[2].shape)
            # else:
            #     print(x.shape)
                
            x=module(x)
            
            self.model.add_module(name, module)#逐层拷贝
            if isinstance(x,tuple):#len=1
                self.model[-1].feature_map=x[0]
                # print(x[0].shape)
            else:
                self.model[-1].feature_map=x#为副本增加feature_map属性，用来存储输出特征图
                # print(x.shape)
            # self.model[-1].feature_map.retain_grad()
            # self.model[-1].feature_map.requires_grad=True
        return x
    
class DeepDream(nn.Module):#
    def __init__(self,model,target_layers,args):
        super().__init__()
        self.pyramid_size=args.pyramid_size
        self.pyramid_ratio=args.pyramid_ratio
        self.gradient_ascent_epochs=args.gradient_ascent_epochs
        self.lr=args.deepdream_lr
        self.smoothing_coefficient=args.smoothing_coefficient
        self.upper_img_bound=torch.tensor(args.upper_img_bound).cuda()
        self.lower_img_bound=torch.tensor(args.lower_img_bound).cuda()
        self.model=model.requires_grad_(True)#对输入图进行优化，不对模型进行优化
        self.aag=ActivationsAndGradients(model=model, target_layers=target_layers, is_channel=args.is_channel,channel=args.channel)

    def forward(self, x):  
        # x.requires_grad=True  #不是zero都需要这个
        # layer_names=[str(i) for i in range(self.start,self.end+1)]

        # #调用自定义ForwardMap类返回特征图
        # layer_names=['22']#['15','18','20','22','25']=['relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1']
        # yaml_path='/home/mdisk1/zhongchulong/interpt/githubs/yolov5/models/yolov5s.yaml'
        # maps=ForwardMap(model=self.model.model.model,cfg=yaml_path)


        original_shape=x.shape[-2:]# save initial height and width


        # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
        # Going from smaller to bigger resolution (from pyramid top to bottom)
        for pyramid_level in range(self.pyramid_size):
            exponent=pyramid_level- self.pyramid_size + 1
            new_shape = np.round(np.float32(original_shape) * (self.pyramid_ratio**exponent)).astype(np.int32)

            x=F.interpolate(x,size=tuple(new_shape))
            x.retain_grad()#计算梯度的时候，只有叶子节点才会保留梯度


            for epoch in tqdm(range(1,self.gradient_ascent_epochs+1),desc='Gradient_Ascent'):
                losses=[]
                

                # #调用自定义ForwardMap类返回特征图
                # final_results=maps(x)
                # for name in layer_names:
                #     feature_map=getattr(maps.model, name).feature_map
                #     feature_map=feature_map[0,...]#单通道优化
                #     loss_component = torch.nn.MSELoss(reduction='mean')(feature_map, torch.zeros_like(feature_map))
                #     losses.append(loss_component)
                
                #调用hook函数返回特征图
                x_outputs = self.aag(x)
                activations_list = [a for a in self.aag.activations]
                for feature_map in activations_list:
                    loss_component = torch.nn.MSELoss(reduction='mean')(feature_map, torch.zeros_like(feature_map))
                    losses.append(loss_component)

                loss = torch.mean(torch.stack(losses))
                
                loss.backward(retain_graph=True)#retain_graph=True https://blog.csdn.net/qq_40349484/article/details/118854059
                # loss.backward()
                

                grad = x.grad.data
                sigma=((epoch+1)/self.gradient_ascent_epochs)*2+self.smoothing_coefficient
                smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well
                smooth_grad=grad
                
                
                #归一化
                g_std = torch.std(smooth_grad)
                g_mean = torch.mean(smooth_grad)
                smooth_grad = smooth_grad - g_mean
                smooth_grad = smooth_grad / g_std

                x.data+=self.lr*smooth_grad
                x.grad.data.zero_()
                x.data = torch.max(torch.min(x, self.upper_img_bound), self.lower_img_bound)


        del activations_list
        del smooth_grad
        del grad
        return x