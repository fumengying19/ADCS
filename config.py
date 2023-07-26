''' Configuration File.
'''

class DATASETS(object):
    def __init__(self):
        self.NUM_TRAIN = 50000
        self.NUM_VAL = 50000 - self.NUM_TRAIN
        self.ROOT = {
            'cifar10': '~/Dataset/cifar10',
            'cifar100': '~/Dataset/cifar100'
        }
        # cifar10, cifar100
        self.data = 'cifar10'
        # cifar10:10, cifar100:100
        self.NUM_CLASS = 10


class ACTIVE_LEARNING(object):
    def __init__(self):
        # self.TRIALS = 10
        # self.CYCLES = 1 # ????
        self.TRIALS = 1 # Performance = np.zeros((3, 10))
        # cifar10 10; cifar100 7
        self.CYCLES = 10 # Performance = np.zeros((3, 10))
        # cifar10 1000; cifar100 5000;
        self.INIT = 1000
        # cifar10 1000; cifar100 2500
        self.ADDENDUM = 1000
        # cifar10 2000; cifar100 5000
        self.NUM = 2000
        self.SELECT = 100
        self.SUBSET = 10000


class TRAIN(object):
    def __init__(self):
        self.BATCH = 128
        self.EPOCH = 200
        # cifar10 0.1; cifar100 0.01
        self.LR = 0.1
        self.MILESTONES = [160]
        self.EPOCHL = 120
        self.MOMENTUM = 0.9
        self.WDECAY = 5e-4
        self.MIN_CLBR = 0.1
        self.MAX_CLBR = 0.1
        self.PATH = './simsiam-cifar10-experiment-resnet18_cifar_variant1_0717130844.pth'


class CONFIG(object):
    def __init__(self, port=9000):
        self.port = port
        self.DATASET = DATASETS()
        self.ACTIVE_LEARNING = ACTIVE_LEARNING()
        self.TRAIN = TRAIN()

