import pyrealsense2 as rs
import argparse
import cv2
import math
from models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform
#cudnn.benchmark = True
cudnn.benchmark = False
test_res = 0.7

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 1280, 800, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 800, rs.format.y8, 30)
profile = pipeline.start(config)

# construct model
model = hsm(128,-1,level=1)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

pretrained_dict = torch.load("./weights/final-768px.tar")
pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
model.load_state_dict(pretrained_dict['state_dict'],strict=False)

model.eval()

processed = get_transform()

try:
    while True:
        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        if not nir_lf_frame or not nir_rg_frame:
            continue
        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image = np.asanyarray(nir_rg_frame.get_data())

        imgL_o = np.stack([nir_lf_image, nir_lf_image, nir_lf_image], axis=2).astype('float32')
        imgR_o = np.stack([nir_rg_image, nir_rg_image, nir_rg_image], axis=2).astype('float32')

        imgsize = imgL_o.shape[:2]

        max_disp = imgsize[1]

        tmpdisp = int(max_disp*test_res//64*64)
        if (max_disp*test_res/64*64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp ==64: model.module.maxdisp=128
        model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
        model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
        model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()

        # resize
        imgL_o = cv2.resize(imgL_o,None,fx=test_res,fy=test_res,interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o,None,fx=test_res,fy=test_res,interpolation=cv2.INTER_CUBIC)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h-imgL.shape[2]
        left_pad = max_w-imgL.shape[3]
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        # test
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())

        with torch.no_grad():
            torch.cuda.synchronize()
            pred_disp,entropy = model(imgL,imgR)
            torch.cuda.synchronize()
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad   = max_h-imgL_o.shape[0]
        left_pad  = max_w-imgL_o.shape[1]
        entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

        # resize to highres
        pred_disp = cv2.resize(pred_disp/test_res,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf

        # horizontal stack
        image = pred_disp/255
        cv2.imshow('IR Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
