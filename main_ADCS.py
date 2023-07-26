import random, time, os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10
import protorype
import config
from resnet_cifar import BasicBlock, Bottleneck, ResNet as ResNet_Cifar
import loader
import torch.distributed as dist
from sklearn.cluster import KMeans

class printlog():
    def __init__(self, filename, sync_terminal=True, mode='w+'):
        folder = os.path.split(os.path.realpath(filename))[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.file = open(filename, mode=mode)
        self.sync_terminal = sync_terminal
        self.gettime = lambda:time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())

    def log(self, *x, withtime=True):
        timestr = f'[{self.gettime()}] =>' if withtime else ''
        print(timestr, *x, file=self.file, flush=True)
        if self.sync_terminal:
            print(timestr, *x)

    def close(self):
        self.file.close()


class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


def get_dataset(dataset='cifar10', root=''):

    cifar = {'cifar10': CIFAR10, 'cifar100': CIFAR100}[dataset]
    mean = {'cifar10': [0.4914, 0.4822, 0.4465], 'cifar100': [0.5071, 0.4867, 0.4408]}[dataset]
    std = {'cifar10': [0.2023, 0.1994, 0.2010], 'cifar100': [0.2675, 0.2565, 0.2761]}[dataset]

    # cifar10
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # augmentation1 = [
    #     transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=1.0),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ]

    augmentation1 = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                            transforms.Compose(augmentation2))
    test_transform = loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                            transforms.Compose(augmentation2))
    # train_transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(size=32, padding=4),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ])
    # test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ])
    train = cifar(root, train=True, download=True, transform=train_transform)
    unlabeled = cifar(root, train=True, download=True, transform=train_transform)
    test = cifar(root, train=False, download=True, transform=test_transform)
    
    return train, test, unlabeled


class metric_entropy(nn.Module):
    def __init__(self):
        super(metric_entropy, self).__init__()

    def forward(self, scores1, scores2):
        scores1 = scores1 / torch.sum(scores1, dim=1).unsqueeze(dim=1).repeat(1, scores1.size(1))
        scores2 = scores2 / torch.sum(scores2, dim=1).unsqueeze(dim=1).repeat(1, scores2.size(1))
        return -torch.sum(scores1*torch.log2(scores1) + scores2*torch.log2(scores2), dim=1) / 2

class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads

class criterion_consist(nn.Module):
    def __init__(self, reduction='none'):
        super(criterion_consist, self).__init__()
        self.criterion = nn.BCELoss(reduction=reduction)

    @staticmethod
    def cosine_distance(feat1, feat2):
        return torch.matmul(F.normalize(feat1), F.normalize(feat2).t())

    @staticmethod
    def sharpen(p):
        sharp_p = p**(1./0.25)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    @staticmethod
    def snn(query, supports, labels, tau=0.1):
        """ Soft Nearest Neighbours similarity classifier """
        softmax = torch.nn.Softmax(dim=1)
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)
        supports = supports.cuda()

        # Step 3: compute similarlity between local embeddings
        return softmax(query @ supports.t() / tau) @ labels


    def forward(self, p1, p2, k1, k2):
        # scoresN: scores tensor given by different nets with the same data batch;
        # labels: labels of protp;
        p1 = p1.squeeze(-1).squeeze(-1)
        p2 = p2.squeeze(-1).squeeze(-1)

        k1 = k1.squeeze(-1).squeeze(-1)
        k2 = k2.squeeze(-1).squeeze(-1)
        l = torch.tensor([0,1,2,3,4,5,6,7,8,9])
        l = l.cuda()

        anchor_views = k1
        anchor_supports = p1
        anchor_support_labels = get_one_hot_label(l, 10)
        target_views = k2
        target_supports = p2
        target_support_labels = get_one_hot_label(l, 10)

        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, anchor_supports, anchor_support_labels, tau=0.1)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.snn(target_views, target_supports, target_support_labels, tau=0.1)
            targets = self.sharpen(targets)
            # if multicrop > 0:
            #     mc_target = 0.5 * (targets[:batch_size] + targets[batch_size:])
            #     targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        avg_probs = AllReduce.apply(torch.mean(self.sharpen(probs), dim=0))
        rloss -= torch.sum(torch.log(avg_probs ** (-avg_probs)))

        return loss + rloss


class criterion_init(nn.Module):
    def __init__(self, reduction='none'):
        super(criterion_init, self).__init__()
        self.criterion = nn.BCELoss(reduction=reduction)

    def forward(self, scores1, scores2, labels):
        # scoresN: scores tensor given by different nets with the same data batch; 
        # labels: labels of this data batch;
        return torch.sum(self.criterion(scores1, labels) + self.criterion(scores2, labels)) / scores1.size(0) 


class criterion_backbone(nn.Module):
    def __init__(self, tao=0.1, alpha=128):
        super(criterion_backbone, self).__init__()
        self.tao = tao
        self.alpha = alpha
        self.metric = metric_entropy()

    def forward(self, scores1, scores2):
        loss = torch.mean(torch.abs(scores1 - scores2), dim=1) # fomular 2
        loss_weight = (1 - torch.sigmoid(self.metric(scores1, scores2) - self.tao)) / self.alpha
        return torch.mean(loss_weight.detach() * loss) # fomular 5


class criterion_classifier(nn.Module):
    # 286.FuM-ADS: Fomular 8, Fomular 6,7
    # train_ADA,py: line 127-138 (batchsize=128)
    def __init__(self, tao=0.1, alpha=128):
        super(criterion_classifier, self).__init__()
        self.tao = tao # tao=0.1
        self.alpha = alpha # train_ADS.py, line 137, alpha=128=batchsize, in 286.FuM-ADS.pdf, alpha should be batchsize, but why batchsize is involved in loss calculation in one graph?  
        self.metric = metric_entropy()

    def forward(self, scores1, scores2):
        loss = 1 - torch.mean(torch.abs(scores1 - scores2), dim=1) # fomular 6
        loss_weight = torch.sigmoid(self.metric(scores1, scores2) - self.tao) / self.alpha # fomular 7
        return torch.mean(loss_weight.detach() * loss) # fomular 8


class ADSNet_backbone(ResNet_Cifar):
    # models/resnet.py: ResNet line 115-120
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(ADSNet_backbone, self).__init__(block=block, num_blocks=num_blocks, num_classes=num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


class ADSNet_classifier(nn.Module):
    # models/resnet.py: ResNet line 122-130
    def __init__(self, alpha=0.05, num_proto=1, num_classes=10, block_expansion=1):
        super(ADSNet_classifier, self).__init__()
        self.alpha = alpha
        self.num_proto = num_proto
        self.num_classes = num_classes
        self.channels = 512*block_expansion
        self.conv  = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.protos = nn.Parameter(torch.randn(self.num_proto * self.num_classes, self.channels), requires_grad=True)

    @staticmethod
    def cosine_distance(feat1, feat2):
        return torch.matmul(F.normalize(feat1), F.normalize(feat2).t())

    def forward(self, x):
        # input x: feature (B,C=512,1,1)
        # before F.avg_pool2d(out, 4) in ResNet in "https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py", feature is (B,C,4,4)
        x = torch.sigmoid(self.conv(x).view(x.size(0), -1))
        cls_score, _ = self.cosine_distance(x, self.protos).view(-1, self.num_proto, self.num_classes).max(dim=1)
        return torch.sigmoid(cls_score / self.alpha)

class ADSNet(nn.Module):
    # models/resnet.py: ResNet
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        # ResNet18: BasicBlock, [2, 2, 2, 2]
        # ResNet34: BasicBlock, [3, 4, 6, 3]
        # ResNet50: Bottleneck, [3, 4, 6, 3]
        super(ADSNet, self).__init__()
        self.backbone = ADSNet_backbone(block=block, num_blocks=num_blocks, num_classes=num_classes)
        self.classifier1 = ADSNet_classifier(num_classes=num_classes, block_expansion=block.expansion)
        self.classifier2 = ADSNet_classifier(num_classes=num_classes, block_expansion=block.expansion)
        self.classifier3 = ADSNet_classifier(num_classes=num_classes, block_expansion=block.expansion)

    def freeze_backbone(self):
        for _, (k, v) in enumerate(self.named_parameters()):
            if k.startswith('backbone'):
                v.requires_grad = False
            else:
                v.requires_grad = True
        params = filter(lambda p: p.requires_grad, self.parameters())   
        return params    

    def freeze_classifier(self):
        ''' ipynb
        ## when classifier is frozen, backbone can still have gradients
        import torch, main_ADS
        net = main_ADS.ADSNet().cuda()
        net.freeze_classifier()
        score1, score2 = net(torch.randn(1,3,32,32).cuda())
        (torch.sum(score1) + torch.sum(score2)).backward()
        for _, (k,v) in enumerate(net.named_parameters()):
            print(k, v.grad)
        '''
        for _, (k, v) in enumerate(self.named_parameters()):
            if k.startswith('classifier'):
                v.requires_grad = False
            else:
                v.requires_grad = True
        params = filter(lambda p: p.requires_grad, self.parameters())   
        return params

    def unfreeze_all(self):
        for _, (_, v) in enumerate(self.named_parameters()):
            v.requires_grad = True
        return self.parameters()

    def forward(self, x):
        out = self.backbone(x)
        score1 = self.classifier1(out)
        score2 = self.classifier2(out)
        score3 = self.classifier3(out)

        return score1, score2, score3, out


def get_one_hot_label(labels=None, num_classes=10):
    # from labels: [0, 1, 2] to labels:[[1,0,0,0],[0,1,0,0],[0,0,1,0]] (Ncls=4)
    return torch.zeros(labels.shape[0], num_classes, device=labels.device).scatter_(1, labels.view(-1, 1), 1)


def train_init(model=ADSNet().cuda(), _criterion_init=criterion_init(), labeled_loader=None, optimizer=None):
    # train backbone and classifier
    model.train()
    model.unfreeze_all()

    loss = 0
    for data in labeled_loader:
        inputs_1 = data[0][0].cuda()
        inputs_2 = data[0][1].cuda()
        labels = get_one_hot_label(data[1]).cuda()

        optimizer.zero_grad()
        scores1, scores2, scores3, out = model(inputs_1)
        scores4, scores5, scores6, out = model(inputs_2)
        loss = (_criterion_init(scores1, scores2, labels) + _criterion_init(scores3, scores3, labels) +
                _criterion_init(scores4, scores5, labels) + _criterion_init(scores6, scores6, labels)) / 2
        loss.backward()
        optimizer.step()

    return loss


def train_backbone(model=ADSNet().cuda(), _criterion_backbone=criterion_backbone(), unlabeled_loader=None, optimizer=None):
    model.train()
    params = model.freeze_classifier()

    loss = 0
    for data in unlabeled_loader:
    
        params = model.freeze_classifier()
        optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        inputs_1 = data[0][0].cuda()
        inputs_2 = data[0][1].cuda()
        scores1, scores2, _, _ = model(inputs_1)
        scores3, scores4, _, _ = model(inputs_2)
        loss = _criterion_backbone(scores1, scores2) + _criterion_backbone(scores3, scores4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    return loss


def train_classifier(model=ADSNet().cuda(), _criterion_init=criterion_init(), _criterion_classifier=criterion_classifier(), labeled_loader=None, unlabeled_loader=None, optimizer=None):
    model.train()
    params = model.freeze_backbone()

    loss = 0
    inputlist = list(enumerate(labeled_loader))
    inputlist_len = len(inputlist)
    for i, (uinputs, _) in enumerate(unlabeled_loader):
        (_, (inputs, labels)) = inputlist[i % inputlist_len]
        inputs_0 = inputs[0].cuda()
        inputs_1 = inputs[1].cuda()
        labels = get_one_hot_label(labels).cuda()

        params = model.freeze_backbone()
        optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        uinputs_0 = uinputs[0].cuda()
        uinputs_1 = uinputs[1].cuda()
        scores1, scores2, _, _ = model(inputs_0)
        scores3, scores4, _, _ = model(inputs_1)
        uscores1, uscores2, _, _ = model(uinputs_0)
        uscores3, uscores4, _, _ = model(uinputs_1)

        loss = _criterion_init(scores1, scores2, labels) + _criterion_classifier(uscores1, uscores2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    return loss


def train_consist(model=ADSNet().cuda(), _criterion_consist=criterion_consist(), labeled_loader=None, unlabeled_loader=None, optimizer=None):
    model.train()
    model.unfreeze_all()

    # get prototype [10,512,1,1]
    cfg = config.CONFIG()
    embedding_list1 = []
    embedding_list2 = []

    label_list = []
    with torch.no_grad():
        for input in labeled_loader:
            data_0 = input[0][0].cuda()
            data_1 = input[0][1].cuda()
            label = input[1].cuda()
            _, _, _, embedding1 = model(data_0)
            _, _, _, embedding2 = model(data_1)

            embedding_list1.append(embedding1.cpu())
            embedding_list2.append(embedding2.cpu())

            label_list.append(label.cpu())
    embedding_list1 = torch.cat(embedding_list1, dim=0)
    embedding_list2 = torch.cat(embedding_list2, dim=0)

    label_list = torch.cat(label_list, dim=0)

    proto_list1 = []
    proto_list2 = []

    for class_index in range(cfg.DATASET.NUM_CLASS):
        data_index = (label_list == class_index).nonzero()
        embedding_this1 = embedding_list1[data_index.squeeze(-1)]
        embedding_this2 = embedding_list2[data_index.squeeze(-1)]

        # one prototype for one class
        embedding_this1 = embedding_this1.mean(0)
        embedding_this2 = embedding_this2.mean(0)
        proto_list1.append(embedding_this1)
        proto_list2.append(embedding_this2)

    proto_list1 = torch.stack(proto_list1, dim=0)
    proto_list2 = torch.stack(proto_list2, dim=0)


    # get loss
    loss = 0
    for i, (uinputs, _) in enumerate(unlabeled_loader):
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        uinputs_0 = uinputs[0].cuda()
        uinputs_1 = uinputs[1].cuda()
        uscores1, uscores2, _, uembedding_0 = model(uinputs_0)
        uscores3, uscores4, _, uembedding_1 = model(uinputs_1)

        loss = _criterion_consist(proto_list1, proto_list2, uembedding_0, uembedding_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    return loss

def train_cycle(cfg, model, train_loader, unlabeled_loader):
    pass


def test_cycle(model, test_loader):
    model.eval()

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs_0 = inputs[0].cuda()
            inputs_1 = inputs[1].cuda()
            labels = labels.cuda()

            scores1, scores2, scores3, out = model(inputs_0)
            _, preds1 = torch.max(scores1.data, 1)
            _, preds2 = torch.max(scores2.data, 1)
            _, preds3 = torch.max(scores3.data, 1)
            total += labels.size(0)
            correct1 += (preds1 == labels).sum().item()
            correct2 += (preds2 == labels).sum().item()
            correct3 += (preds3 == labels).sum().item()
            acc1 = 100 * correct1 / total
            acc2 = 100 * correct2 / total
            acc3 = 100 * correct3 / total
            acc = (100 * correct1 / total + 100 * correct2 / total) * 0.5

    return acc1, acc2, acc, acc3


def get_uncertainty(model, unlabeled_loader):
    model.eval()
    uncertainty = torch.tensor([]).cuda()
    metric = metric_entropy()
    with torch.no_grad():
        for (uinputs, _) in unlabeled_loader:
            uinputs_0 = uinputs[0].cuda()
            scores1, scores2, scores3, out = model(uinputs_0)
            uncertainty = torch.cat((uncertainty, metric(scores1, scores2)), dim=0)
    return uncertainty.cpu()

def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix

def Hausdorff_distance(x, y):  # Input be like (Batch,width,height)
    x = x.float() #[128,512]
    y = y.float() #[100,512]
    distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance [128,100]
    # distance_matrix = cos_similar(x, y)   # cos distance
    value1 = distance_matrix.min(1)[0] # 128

    return value1

def normal(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def get_diversity_2(model, unlabeled_loader, labeled_loader):
    model.eval()
    uncertainty = torch.tensor([]).cuda()
    metric = metric_entropy()

    # get prototype [10,512,1,1]
    cfg = config.CONFIG()
    embedding_list1 = []
    embedding_list2 = []

    label_list = []
    with torch.no_grad():
        for input in labeled_loader:
            data_0 = input[0][0].cuda()
            data_1 = input[0][1].cuda()
            label = input[1].cuda()
            _, _, _, embedding1 = model(data_0)
            _, _, _, embedding2 = model(data_1)

            embedding_list1.append(embedding1.cpu())
            embedding_list2.append(embedding2.cpu())

            label_list.append(label.cpu())
    embedding_list1 = torch.cat(embedding_list1, dim=0)
    embedding_list2 = torch.cat(embedding_list2, dim=0)

    label_list = torch.cat(label_list, dim=0)

    proto_list1 = []
    proto_list2 = []

    for class_index in range(cfg.DATASET.NUM_CLASS):
        data_index = (label_list == class_index).nonzero()
        embedding_this1 = embedding_list1[data_index.squeeze(-1)]
        embedding_this2 = embedding_list2[data_index.squeeze(-1)]

        #  1 prototype for ine class
        # embedding_this1 = embedding_this1.mean(0)
        # embedding_this2 = embedding_this2.mean(0)
        #
        # proto_list1.append(embedding_this1)
        # proto_list2.append(embedding_this2)

        # 10 prototype for one class
        kmeans = KMeans(n_clusters=10)
        embedding_this1 = embedding_this1.squeeze(-1).squeeze(-1)  # 95,512
        embedding_this2 = embedding_this2.squeeze(-1).squeeze(-1)
        y1 = kmeans.fit(embedding_this1)
        y2 = kmeans.fit(embedding_this2)
        embedding_this1 = y1.cluster_centers_  # 10,512
        embedding_this2 = y2.cluster_centers_
        embedding_this1 = torch.from_numpy(embedding_this1)
        embedding_this2 = torch.from_numpy(embedding_this2)
        proto_list1.extend(embedding_this1)
        proto_list2.extend(embedding_this2)

    proto_list1 = torch.stack(proto_list1, dim=0) # proto:[10,512,1,1]
    proto_list2 = torch.stack(proto_list2, dim=0)

    with torch.no_grad():
        for (uinputs, _) in unlabeled_loader:
            uinputs_0 = uinputs[0].cuda()
            uinputs_1 = uinputs[1].cuda()
            # out1:[128,512,1,1], score:[128,10]
            scores1, scores2, scores3, out1 = model(uinputs_0)
            scores4, scores5, scores6, out2 = model(uinputs_1)
            proto_list1 = proto_list1.squeeze(-1).squeeze(-1).cuda()
            proto_list2 = proto_list2.squeeze(-1).squeeze(-1).cuda()
            out1 = out1.squeeze(-1).squeeze(-1)
            out2 = out2.squeeze(-1).squeeze(-1)

            uncertainty = torch.cat((uncertainty, Hausdorff_distance(out1, proto_list1)), dim=0)
    return uncertainty.cpu()


def get_diversity(model, unlabeled_loader, labeled_loader):
    model.eval()
    uncertainty = torch.tensor([]).cuda()
    distance = torch.tensor([]).cuda()
    Out = torch.tensor([]).cuda()
    new_select_arg = torch.tensor([]).cuda()
    new_select_feat = torch.tensor([]).cuda()
    metric = metric_entropy()

    # get prototype [10,512,1,1]
    cfg = config.CONFIG()
    embedding_list1 = []
    embedding_list2 = []

    label_list = []
    with torch.no_grad():
        for input in labeled_loader:
            data_0 = input[0][0].cuda()
            data_1 = input[0][1].cuda()
            label = input[1].cuda()
            _, _, _, embedding1 = model(data_0)
            _, _, _, embedding2 = model(data_1)

            embedding_list1.append(embedding1.cpu())
            embedding_list2.append(embedding2.cpu())

            label_list.append(label.cpu())
    embedding_list1 = torch.cat(embedding_list1, dim=0)
    embedding_list2 = torch.cat(embedding_list2, dim=0)

    label_list = torch.cat(label_list, dim=0)

    proto_list1 = []
    proto_list2 = []

    for class_index in range(cfg.DATASET.NUM_CLASS):
        data_index = (label_list == class_index).nonzero()
        embedding_this1 = embedding_list1[data_index.squeeze(-1)]
        embedding_this2 = embedding_list2[data_index.squeeze(-1)]

        # embedding_this1 = embedding_this1.mean(0)
        # embedding_this2 = embedding_this2.mean(0)
        #
        # proto_list1.append(embedding_this1)
        # proto_list2.append(embedding_this2)

        # 10 prototype for one class
        kmeans = KMeans(n_clusters=10)
        embedding_this1 = embedding_this1.squeeze(-1).squeeze(-1) # 95,512
        embedding_this2 = embedding_this2.squeeze(-1).squeeze(-1)
        y1 = kmeans.fit(embedding_this1)
        y2 = kmeans.fit(embedding_this2)
        embedding_this1 = y1.cluster_centers_  # 10,512
        embedding_this2 = y2.cluster_centers_
        embedding_this1 = torch.from_numpy(embedding_this1)
        embedding_this2 = torch.from_numpy(embedding_this2)
        proto_list1.extend(embedding_this1)
        proto_list2.extend(embedding_this2)

    proto_list1 = torch.stack(proto_list1, dim=0) # proto:[100,512]
    proto_list2 = torch.stack(proto_list2, dim=0)

    with torch.no_grad():
        for (uinputs, _) in unlabeled_loader:
            uinputs_0 = uinputs[0].cuda()
            uinputs_1 = uinputs[1].cuda()
            # out1:[128,512,1,1], score:[128,10]
            scores1, scores2, scores3, out1 = model(uinputs_0)
            scores4, scores5, scores6, out2 = model(uinputs_1)
            # proto_list1 = proto_list1.squeeze(-1).squeeze(-1).cuda()
            # proto_list2 = proto_list2.squeeze(-1).squeeze(-1).cuda()
            proto_list1 = proto_list1.cuda()
            proto_list2 = proto_list2.cuda()
            out1 = out1.squeeze(-1).squeeze(-1)
            out2 = out2.squeeze(-1).squeeze(-1)

            Out = torch.cat((Out, out1),dim=0)  # [10000,512]
            distance = torch.cat((distance, Hausdorff_distance(out1, proto_list1)), dim=0)  # [10000], [0.0036--0.0512]
        # s_0 = np.argsort(distance.cpu())[:cfg.ACTIVE_LEARNING.SELECT]
        # un_remain = np.argsort(distance.cpu())[cfg.ACTIVE_LEARNING.SELECT:]  # [9900]
        s_0 = np.argsort(distance.cpu())[-cfg.ACTIVE_LEARNING.SELECT:]
        un_remain = np.argsort(distance.cpu())[: -cfg.ACTIVE_LEARNING.SELECT]  # [9900]
        new_select_arg = torch.cat((new_select_arg, s_0.float().cuda()), dim=0)  # [100]
        new_select_feat = torch.cat((new_select_feat, Out[s_0]), dim=0)  # [100,512]
        for i in range(9):
            dist_un_ns = Hausdorff_distance(Out, new_select_feat)  # [10000], [0--0.0504]
            # normalization
            d1 = normal(distance.cpu().numpy())  # [0--1]
            d2 = normal(dist_un_ns.cpu().numpy())  # [0--1]
            new_dist = torch.tensor(d1 * d2)  #  [10000], [0--0.2167]
            # s_new = np.argsort(new_dist)[:cfg.ACTIVE_LEARNING.SELECT]
            s_new = np.argsort(new_dist)[-cfg.ACTIVE_LEARNING.SELECT:]
            un_remain = list(set(un_remain.tolist())-set(s_new.tolist()))
            un_remain = torch.tensor(un_remain)
            new_select_arg = torch.cat((new_select_arg, s_new.float().cuda()), dim=0)
            new_select_feat = torch.cat((new_select_feat, Out[s_new]), dim=0)

    return list(new_select_arg.int().cpu().numpy()), list(un_remain.numpy())
            # uncertainty = torch.cat((uncertainty, Hausdorff_distance(out1, proto_list1)), dim=0)
    # return uncertainty.cpu()


def train_ADS(dataset='cifar10'):
    cfg = config.CONFIG()
    checkpoint_dir =  os.path.realpath(os.path.join('./output', time.strftime('%m%d%H%M%S', time.localtime())))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log = printlog(checkpoint_dir + '/log.log')

    for k, v in cfg.__dict__.items():
        try:
           for k1, v1 in v.__dict__.items(): 
               log.log(f'{k1}:{v1}')
        except:
            log.log(f'{k}:{v}')
    log.log(f'checkpoint:{checkpoint_dir}')
        
    train_dataset, test_dataset, unlabeled_dataset = get_dataset(dataset, cfg.DATASET.ROOT[dataset])
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH)

    Performance = np.zeros((3, 10))
    log.log('Train Start.')
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):

        torch.backends.cudnn.benchmark = True
        model = ADSNet(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10).cuda()

        indices = list(range(cfg.DATASET.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cfg.ACTIVE_LEARNING.INIT]
        unlabeled_set = indices[cfg.ACTIVE_LEARNING.INIT:]
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)


        for cycle in range(cfg.ACTIVE_LEARNING.CYCLES):
            # model = ADSNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10).cuda()
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetSequentialSampler(subset), pin_memory=True)

            optim_init = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            optim_backbone = optim.SGD(model.freeze_classifier(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            optim_classifier = optim.SGD(model.freeze_backbone(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            scheduler = lr_scheduler.MultiStepLR(optim_init, milestones=cfg.TRAIN.MILESTONES)

            _criterion_init=criterion_init()
            _criterion_backbone=criterion_backbone()
            _criterion_classifier=criterion_classifier()
            _criterion_consist=criterion_consist()

            loss_consist = 0

            for epoch in range(cfg.TRAIN.EPOCH):
                if epoch == 0:
                    train_init(model=model, _criterion_init=_criterion_init, labeled_loader=train_loader, optimizer=optim_init)

                if epoch >= 50:
                    loss_consist = train_consist(model=model, _criterion_consist=_criterion_consist, labeled_loader=train_loader, unlabeled_loader=unlabeled_loader, optimizer=optim_init)

                loss_b = train_backbone(model=model, _criterion_backbone=_criterion_backbone, unlabeled_loader=unlabeled_loader, optimizer=optim_backbone)
                loss_c = train_classifier(model=model, _criterion_init=_criterion_init, _criterion_classifier=_criterion_classifier, labeled_loader=train_loader, unlabeled_loader=unlabeled_loader, optimizer=optim_classifier)
                loss_i = train_init(model=model, _criterion_init=_criterion_init, labeled_loader=train_loader, optimizer=optim_init)

                if epoch == 50:
                    model = protorype.replace_base_fc(train_loader, model)

                scheduler.step()

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    log.log(f'Trial [{trial + 1}/{cfg.ACTIVE_LEARNING.TRIALS}]',
                        f'Cycle [{cycle + 1}/{cfg.ACTIVE_LEARNING.CYCLES}]',
                        f'Epoch [{epoch + 1}/{cfg.TRAIN.EPOCH}]',
                        f'Labeled Set Size [{len(labeled_set)}]',
                        f'Loss [backbone: {loss_b:.3g}, Loss_classifier: {loss_c:.3g}, Loss_init: {loss_i:.3g}, Loss_consist:{loss_consist:.3g}]'
                    )

            acc1, acc2, acc, acc3 = test_cycle(model, test_loader)
            Performance[trial, cycle] = acc

            log.log(f'Trial [{trial + 1}/{cfg.ACTIVE_LEARNING.TRIALS}]',
                        f'Cycle [{cycle + 1}/{cfg.ACTIVE_LEARNING.CYCLES}]',
                        f'Labeled Set Size [{len(labeled_set)}]',
                        f'Test [acc1: {acc1:.3g}, acc2: {acc2:.3g}, acc: {acc:.3g}, acc3:{acc3:.3g}]'
            )

            #  uncertainty
            arg = np.argsort(get_uncertainty(model, unlabeled_loader))  # small----big
            new_set = list(torch.tensor(subset)[arg][-cfg.ACTIVE_LEARNING.NUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-cfg.ACTIVE_LEARNING.NUM].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]
            new_unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetSequentialSampler(new_set), pin_memory=True)

            #  diversity1
            # select_arg, unlabeled_arg = get_diversity(model, new_unlabeled_loader, train_loader)
            # labeled_set += np.array(new_set)[select_arg].tolist()
            # unlabeled_set = np.array(new_set)[unlabeled_arg].tolist() + unlabeled_set
            # diversity2
            arg2 = np.argsort(get_diversity_2(model, new_unlabeled_loader, train_loader))
            select_arg = arg2[-cfg.ACTIVE_LEARNING.ADDENDUM:]
            unlabeled_arg = arg2[:-cfg.ACTIVE_LEARNING.ADDENDUM]
            labeled_set += np.array(new_set)[select_arg].tolist()
            unlabeled_set = np.array(new_set)[unlabeled_arg].tolist() + unlabeled_set

            # unlabeled_set = list(set(arg.tolist())-set(labeled_set))
            # arg = np.argsort(get_uncertainty(model, unlabeled_loader, train_loader))  # small----big
            # labeled_set += list(torch.tensor(subset)[arg][-cfg.ACTIVE_LEARNING.ADDENDUM:].numpy())
            # unlabeled_set = list(torch.tensor(subset)[arg][:-cfg.ACTIVE_LEARNING.ADDENDUM].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]
            train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

            torch.save(model.state_dict(), f'{checkpoint_dir}/ADSNet_trial_{trial}_cycle_{cycle}_epoch_{cfg.TRAIN.EPOCH}.pth')

    log.log('Performance Summary: ', withtime=False)
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):
        log.log(f'Trail {trial + 1}: {Performance[trial]}', withtime=False)
    
    log.close()


if __name__ == '__main__':
    train_ADS()
