# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging 
import os
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width ,boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
       
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img , axis=(0,1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, 
                cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                round(s_x), self.channel_average)
       
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
            
        # scale penalty
        s_c = change(sz(pred_bbox[2,:], pred_bbox[3,:]) / 
                (sz(self.size[0]*scale_z, self.size[1]*scale_z))) 

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2,:]/pred_bbox[3,:]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty 
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
   
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }


class BatchSiamRPNTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        bs = delta.shape[0]
        delta = delta.view(bs, 4, -1)
        delta = delta.data.cpu().numpy()

        delta[:, 0, :] = delta[:, 0, :] * anchor[None, :, 2] + anchor[None, :, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[None, :, 3] + anchor[None, :, 1]
        delta[:, 2, :] = np.exp(delta[:, 2, :]) * anchor[None, :, 2]
        delta[:, 3, :] = np.exp(delta[:, 3, :]) * anchor[None, :, 3]
        return delta

    def _convert_score(self, score):
        bs = score.shape[0]
        score = score.view(bs, 2, -1)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = np.clip(cx, 0, boundary[1])
        cy = np.clip(cy, 0, boundary[0])
        width = np.minimum(width, boundary[1])
        height = np.minimum(height, boundary[0])
        width = np.maximum(10, width)
        height = np.maximum(10, height)
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: bs x 4 [x, y, w, h] bbox
        """
        bs = bbox.shape[0]
        self.center_pos = np.empty((bs, 2))
        self.size = np.empty((bs, 2))
        center_x = bbox[:, 0] + (bbox[:, 2] - 1) / 2.0
        center_y = bbox[:, 1] + (bbox[:, 3] - 1) / 2.0
        self.center_pos[:, 0] = center_x
        self.center_pos[:, 1] = center_y
        self.size[:, 0] = bbox[:, 2]
        self.size[:, 1] = bbox[:, 3]
       
        # calculate z crop size
        w_z = self.size[:, 0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, axis=1)
        h_z = self.size[:, 1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, axis=1)
        s_z = np.sqrt(w_z * h_z)

        # calculate channle average
        self.channel_average = np.mean(img , axis=(0,1))

        # get crop
        z_crop_list = []
        for i in range(bs):
            z_crop = self.get_subwindow(
                img, 
                self.center_pos[i],
                cfg.TRACK.EXEMPLAR_SIZE, 
                round(s_z[i]), 
                self.channel_average)
            z_crop_list.append(z_crop)
        z_crop = torch.cat(z_crop_list, 0)
        self.model.template(z_crop)


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        bs = self.size.shape[0]
        w_z = self.size[:, 0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, axis=1)
        h_z = self.size[:, 1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size, axis=1)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop_list = []
        for i in range(bs):
            x_crop = self.get_subwindow(
                img, 
                self.center_pos[i], 
                cfg.TRACK.INSTANCE_SIZE,
                round(s_x[i]), 
                self.channel_average)
            x_crop_list.append(x_crop)
        x_crop = torch.cat(x_crop_list, 0)
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
            
        # scale penalty
        s_c = change(sz(pred_bbox[:, 2,:], pred_bbox[:, 3,:]) / 
                (sz(self.size[:, 0]*scale_z, self.size[:, 1]*scale_z))[:, None]) 

        # aspect ratio penalty
        r_c = change((self.size[:, 0]/self.size[:, 1])[:, None] /
                     (pred_bbox[:, 2, :]/pred_bbox[:, 3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty 
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore, axis=1)

        bbox = pred_bbox[range(bs), :, best_idx] / scale_z[:, None]
        lr = penalty[range(bs), best_idx] * score[range(bs), best_idx] * cfg.TRACK.LR
        
        cx = bbox[:, 0] + self.center_pos[:, 0]
        cy = bbox[:, 1] + self.center_pos[:, 1]

        # smooth bbox
        width = self.size[:, 0] * (1 - lr) + bbox[:, 2] * lr
        height = self.size[:, 1] * (1 - lr) + bbox[:, 3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos[:, 0] = cx
        self.center_pos[:, 1] = cy
        self.size[:, 0] = width
        self.size[:, 1] = height

        bbox = np.empty((bs, 4))
        bbox[:, 0] = cx - width / 2
        bbox[:, 1] = cy - height / 2
        bbox[:, 2] = width
        bbox[:, 3] = height
   
        best_score = score[range(bs), best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }