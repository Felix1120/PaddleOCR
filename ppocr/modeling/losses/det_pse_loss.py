# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from paddle import fluid
from .det_basic_loss import BalanceLoss, DiceLoss


class PSELoss(object):
    """
    PSE Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self, params):
        super(PSELoss, self).__init__()
        self.balance_loss = params['balance_loss']
        self.main_loss_type = params['main_loss_type']
        self.ohem_ratio = params['ohem_ratio']
        self.Lambda = params['Lambda']

    def __call__(self, predicts, labels):
        texts = predicts['maps'][:, -1, :, :]
        kernels = predicts['maps'][:, :-1, :, :]
        gt_texts = labels['shrink_maps'][:, -1, :, :]
        gt_kernels = labels['shrink_maps'][:, :-1, :, :]
        masks = labels['masks']

        loss_texts = BalanceLoss(
            texts,
            gt_texts,
            masks,
            balance_loss=self.balance_loss,
            main_loss_type=self.main_loss_type,
            negative_ratio=self.ohem_ratio)
        selected_mask = fluid.layers.logical_and((texts > 0.5), (masks > 0.5)).astype(np.float32)

        c = kernels.shape[1]
        new_masks = []
        for _ in range(c):
            new_masks.append(selected_mask)
        new_masks = fluid.layers.stack(new_masks,axis=1)
        # 下面的处理会导致最小map学不到
        # selected_mask = fluid.layers.unsqueeze(selected_mask, axes=1)
        # new_masks = fluid.layers.expand(selected_mask, expand_times=[1, c, 1, 1])
        loss_kernels = DiceLoss(kernels, gt_kernels, new_masks)
        loss_texts = self.Lambda * loss_texts
        loss_kernels = (1 - self.Lambda) * loss_kernels

        loss_all = loss_texts + loss_kernels
        losses = {'total_loss': loss_all,
                  "loss_texts": loss_texts,
                  "loss_kernels": loss_kernels}
        return losses
