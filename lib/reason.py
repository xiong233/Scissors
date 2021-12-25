#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import json
import glob
import os
import  numpy as np

from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.config import cfg

import pdb

class reason(nn.Module):

    def __init__(self, n_classes):
        super(reason, self).__init__()

        self.n_classes = n_classes # 类别个数
        self.lamda = 0.6 #传播范围系数
        self.feature_num = 4096

        self.train_A = nn.Linear(self.feature_num, self.feature_num)
        self.weight_A = -1
        #self.old_A = 10000*torch.ones(4096, 4096).cuda()#self.weight_A = nn.Parameter(torch.zeros(self.feature_num, self.feature_num).cuda())


    def forward(self, rois, pooled_feat):

        batch_size = rois.size(0)
        region_num = rois.size(1)
        fea_num = 4096
        #pdb.set_trace()
        # weight11 = 3000.0 / (pooled_feat.sum(1) + 0.00001).unsqueeze(1).expand(region_num*batch_size, fea_num)
        # norm_pooled_feat = pooled_feat.mul(weight11)
        pooled_feat_view = pooled_feat.view(batch_size, region_num, -1)

        #pooled_feat_view = F.sigmoid(pooled_feat_view)

        raw_region_box = torch.cuda.FloatTensor(rois.data[:,:,1:5])
        # 将每个区域的坐标进行转换
        trs_region_box = self.__transform_box(raw_region_box)#raw_region_box)

        temp0 = torch.zeros(batch_size, region_num, fea_num).cuda()

        for i in range(batch_size):

            A_spread = self.__Build_spread(trs_region_box[i, :, :], region_num)#, nms_mask)
            #pdb.set_trace()

            pooledA = pooled_feat_view[i].sum(0)

            weight11 = 1.0 / (pooledA.sum(0) + 0.00001)#.unsqueeze(1).expand(fea_num, fea_num)
            new_A = pooledA.mul(weight11)
            self.weight_A = 4096 * torch.mm(new_A.unsqueeze(1), new_A.unsqueeze(0))

            reason_fea = self.reason_module(pooled_feat_view[i])
            spread_reason_fea = torch.mm(A_spread, reason_fea)
            #pdb.set_trace()

            temp0[i, :, :] = temp0[i, :, :] + spread_reason_fea

        view2_temp3 = temp0.view_as(pooled_feat)
        return view2_temp3

    def reason_module(self, pooled_fea):

        reason_weight1 = self.train_A(pooled_fea)
        reason_weight2 = torch.mm(pooled_fea, self.weight_A)
        reason_weight = reason_weight1+reason_weight2

        #pdb.set_trace()
        # print ('aaa')
        # print(reason_weight1)
        # print(reason_weight2)
        # print (reason_weight)
        # print (pooled_fea)

        return reason_weight


    def __Build_spread(self, region_box, region_num):

        # 坐标扩展为3维,并转置，用于计算区域之间的距离
        expand1_region_box = region_box.unsqueeze(2)
        expand_region_box = expand1_region_box.expand(region_num, 4, region_num)
        transpose_region_box = expand_region_box.transpose(0,2)

        # 跟据每个区域的w和h，计算每个区域的传播范围
        spread_distance = torch.sqrt(torch.pow(region_box[:, 2], 2) + torch.pow(region_box[:, 3], 2))
        expand_spread_distance = spread_distance.expand(region_num, region_num)

        # 根据每个区域的x和y，计算区域之间的距离
        region_distance = torch.sqrt(torch.pow((expand_region_box[:, 0, :] - transpose_region_box[:, 0, :]), 2) + torch.pow(
            (expand_region_box[:, 1, :] - transpose_region_box[:, 1, :]), 2))

        # A = F.relu(expand_spread_distance-region_distance)

        # 根据传播范围和距离，计算传播矩阵的权值
        A = F.relu(1-region_distance/(self.lamda*expand_spread_distance))
        #A = F.relu(1 - (region_distance-0.3*expand_spread_distance) / (self.lamda*expand_spread_distance))
        # 不接受来自自己的推理信息
        self_w = 1 - torch.eye(region_num, region_num).cuda()

        A = A.mul(self_w)#.mul(nms_w)

        weight = 1.0/(A.sum(1)+0.0001).unsqueeze(1).expand(region_num, region_num)
        #pdb.set_trace()
        return A.mul(weight)

    def __transform_box(self, region_box):

        new_region_box = torch.zeros((region_box.size(0), region_box.size(1), 4)).cuda()
        new_region_box[:,:,0] = 0.5*(region_box[:, :, 0]+region_box[:, :, 2])
        new_region_box[:,:,1] = 0.5*(region_box[:, :, 1]+region_box[:, :, 3])
        new_region_box[:,:,2] = region_box[:, :, 2] - region_box[:, :, 0]
        new_region_box[:,:,3] = region_box[:, :, 3] - region_box[:, :, 1]

        return new_region_box
