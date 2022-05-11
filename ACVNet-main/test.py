# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'# -1 하면 cpu됨

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
# parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
# 메모리사용 128도 18568? 까지는 간다
parser.add_argument('--maxdisp', type=int, default=256, help='maximum disparity')

### 여기 고핌
parser.add_argument('--dataset', default='ourdata', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/home/h/Desktop/left_right", help='data path')
parser.add_argument('--testlist',default='./filenames/ourdata_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--attention_weights_only', default=False, type=str,  help='only train attention weights')
parser.add_argument('--freeze_attention_weights', default=False, type=str,  help='freeze attention weights parameters')
parser.add_argument('--loadckpt', default='./pretrained_model/pretrained_model_sceneflow.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--savepath', default="./ourdata_testoutput/", help='data path')


# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# model, ckpt
model = __models__[args.model](args.maxdisp, args.attention_weights_only, args.freeze_attention_weights)
del args.maxdisp, args.attention_weights_only, args.freeze_attention_weights

model = nn.DataParallel(model)
model.cuda()
state_dict = torch.load(args.loadckpt)
del args.loadckpt

model_dict = model.state_dict()
pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
model_dict.update(pre_dict) 
model.load_state_dict(model_dict)
model.eval()

gc.collect()
torch.cuda.empty_cache()
# del model

# dataset
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

def draw_disparity(disparity_map):

	disparity_map = disparity_map.astype(np.uint8)
	norm_disparity_map = (255*((disparity_map-np.min(disparity_map))/(np.max(disparity_map) - np.min(disparity_map))))
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

def test():
    for batch_idx, sample in enumerate(TestImgLoader):
        gc.collect()
        torch.cuda.empty_cache()

        imgL, imgR = sample['left'], sample['right']
        
        imgL = imgL.cuda()
        imgR = imgR.cuda()

        gc.collect()
        torch.cuda.empty_cache()
        op=model(imgL, imgR)#list
         # [pred_attention, pred0, pred1, pred2]
        
        output=op[0]
        output=output.cpu().detach().numpy() #<class 'numpy.ndarray'>
        output = np.array(output[0])
        color_disparity=draw_disparity(output)
        cv2.imwrite(args.savepath+str(time.time())+'.png',color_disparity)
        output.tofile(args.savepath+str(time.time())+'.raw')
        del imgL, imgR, op, output, color_disparity
        gc.collect()
        torch.cuda.empty_cache()

if __name__=='__main__':
    test()
