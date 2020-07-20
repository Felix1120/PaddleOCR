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

import math

import paddle.fluid as fluid


class PSEHead(object):
    """
    PSE for text detection:
        see https://arxiv.org/abs/1903.12473
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, params):
        self.out_channels = params['out_channels']
        self.inner_channels = params['inner_channels']
        self.C, self.H, self.W = params['image_shape']
        print(self.C, self.H, self.W)

    def __call__(self, conv_features, mode="train"):
        c2, c3, c4, c5 = conv_features
        param_attr = fluid.initializer.MSRAInitializer(uniform=False)
        in5 = fluid.layers.conv2d(
            input=c5,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in4 = fluid.layers.conv2d(
            input=c4,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in3 = fluid.layers.conv2d(
            input=c3,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in2 = fluid.layers.conv2d(
            input=c2,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)

        out4 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=in5, scale=2), y=in4)  # 1/16
        out3 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out4, scale=2), y=in3)  # 1/8
        out2 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out3, scale=2), y=in2)  # 1/4

        p5 = fluid.layers.conv2d(
            input=in5,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p5 = fluid.layers.resize_nearest(input=p5, scale=8)
        p4 = fluid.layers.conv2d(
            input=out4,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p4 = fluid.layers.resize_nearest(input=p4, scale=4)
        p3 = fluid.layers.conv2d(
            input=out3,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p3 = fluid.layers.resize_nearest(input=p3, scale=2)
        p2 = fluid.layers.conv2d(
            input=out2,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)

        fuse = fluid.layers.concat(input=[p5, p4, p3, p2], axis=1)

        # 卷积输出特征图
        y = fluid.layers.conv2d(
            input=fuse,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1
        )
        y = fluid.layers.batch_norm(
            input=y,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu"
        )
        y = fluid.layers.conv2d(
            input=y,
            num_filters=self.out_channels,
            filter_size=1
        )
        y = fluid.layers.resize_nearest(input=y, scale=4)
        if mode != 'train':
            y = fluid.layers.sigmoid(y)
        predicts = {}
        predicts['maps'] = y
        return predicts
