import chainer
from chainer import optimizers
import nutszebra_basic_print


class Optimizer(object):

    def __init__(self, model=None):
        self.model = model
        self.optimizer = None

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()


class OptimizerDense(Optimizer):

    def __init__(self, model=None, schedule=(150, 225), lr=0.1, momentum=0.9, weight_decay=1.0e-4):
        super(OptimizerDense, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerWideRes(Optimizer):

    def __init__(self, model=None, schedule=(60, 120, 160), lr=0.1, momentum=0.9, weight_decay=5.0e-4):
        super(OptimizerWideRes, self).__init__(model)
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i in self.schedule:
            lr = self.optimizer.lr * 0.2
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr


class OptimizerGooglenetV3(Optimizer):

    def __init__(self, model=None, lr=0.045, decay=0.9, weight_decay=1.0e-6, clip=2.0):
        super(OptimizerGooglenetV3, self).__init__(model)
        optimizer = optimizers.RMSprop(lr, decay)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        clip = chainer.optimizer.GradientClipping(clip)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        optimizer.add_hook(clip)
        self.optimizer = optimizer

    def __call__(self, i):
        if i % 2 == 0:
            lr = self.optimizer.lr * 0.94
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
