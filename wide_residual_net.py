import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class WideResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, n=13):
        super(WideResBlock, self).__init__()
        modules = []
        modules += [('bn_relu_conv1_1', BN_ReLU_Conv(in_channel, out_channel))]
        modules += [('bn_relu_conv2_1', BN_ReLU_Conv(out_channel, out_channel))]
        for i in six.moves.range(2, n + 1):
            modules.append(('bn_relu_conv1_{}'.format(i), BN_ReLU_Conv(out_channel, out_channel)))
            modules.append(('bn_relu_conv2_{}'.format(i), BN_ReLU_Conv(out_channel, out_channel)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.n = n

    def weight_initialization(self):
        for i in six.moves.range(1, self.n + 1):
            self['bn_relu_conv1_{}'.format(i)].weight_initialization()
            self['bn_relu_conv2_{}'.format(i)].weight_initialization()

    @staticmethod
    def concatenate_zero_pad(x, h):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h.data.shape
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=h.volatile)
        if type(h.data) is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def __call__(self, x, train=False):
        h = self['bn_relu_conv1_1'](x, train=train)
        h = self['bn_relu_conv2_1'](h, train=train)
        x = h + WideResBlock.concatenate_zero_pad(x, h)
        for i in six.moves.range(2, self.n + 1):
            h = self['bn_relu_conv1_{}'.format(i)](x, train=train)
            x = self['bn_relu_conv2_{}'.format(i)](h, train=train) + x
        return x

    def count_parameters(self):
        count = 0
        for i in six.moves.range(1, self.n + 1):
            count = count + self['bn_relu_conv1_{}'.format(i)].count_parameters()
            count = count + self['bn_relu_conv2_{}'.format(i)].count_parameters()
        return count


class WideResidualNetwork(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=3, out_channels=(16 * 4, 32 * 4, 64 * 4), N=(13, 13, 13)):
        super(WideResidualNetwork, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, 16, 3, 1, 1))]
        in_channel = 16
        for i, out_channel, n in six.moves.zip(six.moves.range(1, block_num + 1), out_channels, N):
            modules.append(('wide_res_block{}'.format(i), WideResBlock(in_channel, out_channel, n=n)))
            in_channel = out_channel
        modules.append(('bn_relu_conv', BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        modules.append(('bn', L.BatchNormalization(category_num)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.block_num = block_num
        self.out_channels = out_channels
        self.N = N
        self.name = 'wide_residual_network_{}_{}_{}_{}'.format(category_num, block_num, out_channels, N)

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        for i in six.moves.range(1, self.block_num + 1):
            self['wide_res_block{}'.format(i)].weight_initialization()
        self.bn_relu_conv.weight_initialization()

    def __call__(self, x, train=False):
        h = self.conv1(x)
        for i in six.moves.range(1, self.block_num + 1):
            h = self['wide_res_block{}'.format(i)](h, train=train)
            h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = F.relu(self.bn(self.bn_relu_conv(h, train=train), test=not train))
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        for i in six.moves.range(1, self.block_num + 1):
            count = count + self['wide_res_block{}'.format(i)].count_parameters()
        count += self.bn_relu_conv.count_parameters()
        return count

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
