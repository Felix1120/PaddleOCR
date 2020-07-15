import subprocess
import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


class PSEPostProcess(object):
    def __init__(self, params):
        self.scale = params['scale']
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.min_size = 3

    def boxes_from_bitmap(self, score, kernels):
        # def pse_warpper(kernals, min_area=5):
        '''
        reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
        :param kernals:
        :param min_area:
        :return:
        '''
        from .pse import pse_cpp
        kernal_num = len(kernels)
        if not kernal_num:
            return np.array([]), []
        label_num, label = cv2.connectedComponents(kernels[0].astype(np.uint8), connectivity=4)
        label_values = []
        for label_idx in range(1, label_num):
            if np.sum(label == label_idx) < self.min_size:
                label[label == label_idx] = 0
                continue
            label_values.append(label_idx)
        pred = pse_cpp(label, kernels, c=kernal_num)
        pred = np.array(pred)
        bbox_list = []
        score_list = []
        for label_value in label_values:
            points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < 800 / (self.scale * self.scale):
                continue

            score_i = np.mean(score[pred == label_value])
            if score_i < self.box_thresh:
                continue
            score_list.append(score_i)
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
        return np.array(bbox_list),np.array(score_list)

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']
        scores = pred[:, -1, :, :]
        kernels = pred > self.thresh
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(scores[batch_index], kernels[batch_index])
            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch
