'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc
import numpy as np
import pickle
from PIL import Image
import cv2

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

# face_mean = (0.5, 0.5, 0.5)
# face_std = (0.5, 0.5, 0.5)

face_mean = (0.5, 0.5, 0.5)
# face_std = (0.5, 0.5, 0.5)
face_std = (0.0039215, 0.0039215, 0.0039215)
# cifar10_mean = (0., 0., 0.)
# cifar10_std = (1., 1., 1.)

mu = torch.tensor(face_mean).view(3, 1, 1).cuda()
std = torch.tensor(face_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

# upper_limit = ((128 - mu) / std)
# lower_limit = ((-128 - mu) / std)

from models import *

def save_img(root, name, img):
    if not os.path.exists(root):
        os.makedirs(root)
    img = img.cpu().numpy()
    img += 127.5
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    cv2.imwrite(os.path.join(root, '%d.jpg' % (name)), img)
    return

class on_attack():
    def __init__(self, g_running):
        self.encoder = g_running.encoder
        self.decoder = g_running.generator
        self.style = g_running.style
        self.penalty = torch.nn.CosineSimilarity().cuda()
        self.noise = []
        self.alpha = 0.5
        # for i in range(0, 6):
        #     size = 4 * 2 ** i
        #     self.noise.append(torch.zeros(1, 1, size, size, requires_grad=False).cuda())

    def apply(self, net, img, targets, eps, thre, iteration=8, alpha=20., t=True):
        img = img.detach()
        targets = net(targets).detach()
        img = transforms.Resize(128)(img)/127.5
        x, m, v = self.encoder(F.avg_pool2d(img, 4))
        x, m, v = x.detach(), m.detach(), v.detach()
        b = img.size(0)
        v = torch.exp(v * 0.5)
        ten = Variable(torch.zeros(b, 512, 4, 4).cuda(), requires_grad=True)
        # t = torch.ones(img.size(0)).float().cuda()
        # final_images = torch.zeros_like(img)

        for i in range(iteration):
            # z = ten * v + m
            rec_img = self.decoder(self.style, x.detach(), ten, v, (-1, -1), attack=True)
            # rec_img = self.alpha * img + (1 - self.alpha) * rec_img
            # rec_img = clamp(rec_img, img - eps, img + eps)
            rec_img = clamp(rec_img, lower_limit, upper_limit)
            # x, _, _ = self.encoder(F.avg_pool2d(rec_img.detach(), 4))
            rec_img = transforms.Resize(224)(rec_img)*127.5
            fea = net(rec_img)
            cos_loss = self.penalty(fea, targets)
            # print(cos_loss)
            if t:
                cos_loss = cos_loss[cos_loss >= thre]
            else:
                cos_loss = cos_loss[torch.where(cos_loss < thre)]
            cos_loss = cos_loss.mean()
            net.zero_grad()
            if ten.grad is not None:
                ten.grad.data.fill_(0)
            cos_loss.backward()
            if t:
                ten = ten - 60. * ten.grad
            else:
                ten = ten + 160. * ten.grad
            ten = Variable(ten.data, requires_grad=True)
            # print('iter%d, cos_loss:%.4f. ' % (i, cos_loss))
        return rec_img, ten.data  # , kl

def recon_w(generator, real_image):
    real_image = transforms.Resize(128)(real_image) / 127.5
    img = F.avg_pool2d(real_image, 4)
    _, _, rec_images = generator(img)
    final_images = transforms.Resize(224)(rec_images) * 127.5
    return final_images

def recon(iter, generator, real_image):
    penalty = torch.nn.MSELoss(reduction='none')
    real_image = transforms.Resize(128)(real_image) / 127.5
    img = F.avg_pool2d(real_image, 4)
    upper = 1e3 * torch.ones(real_image.size(0)).float().cuda()
    final_images = torch.zeros_like(real_image)
    with torch.no_grad():
        for k in range(iter):
            m, v, rec_images = generator(img)
            cost = penalty(rec_images, real_image).mean((1, 2, 3))
            index = cost < upper
            final_images[index] = rec_images.data[index]
            upper = torch.where(index, cost, upper)
    final_images = transforms.Resize(224)(final_images) * 127.5
    return final_images

def loadModel(net, checkpoint=None):
    net_state = net.state_dict()
    load_state = {k: v for k, v in checkpoint.items() if 'fc' not in k}
    # print(load_state.keys())
    net_state.update(load_state)
    net.load_state_dict(net_state)
    # net = resnet50(num_classes=10575, include_top=True)
    # net.load_state_dict(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net = net.eval()
    return net.eval(), device

def load_state_dict(model, fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f)
    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                if('fc.weight'==name or 'fc.bias'==name):
                    # print(name)
                    continue
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

from torchvision import transforms

def get_image(path, img, transformer):
    img = Image.open(os.path.join(path, img))
    img = transformer(img)
    img = img.cuda()
    return img

def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr = optimizer.param_groups[0]['lr'] * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param targets: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = targets.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, targets)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, targets), 1)

        return loss

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def label_smoothing(y_batch_tensor, num_classes, delta):
    y_batch_smooth = (1 - delta - delta / (num_classes - 1)) * \
        y_batch_tensor + delta / (num_classes - 1)
    return y_batch_smooth

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def where(cond, x, y):
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def fgsm_w(net, g, x, targets, eps, alpha, iteration, t = False):
    loss = torch.nn.CosineSimilarity().cuda()
    targets = net(targets).detach().unsqueeze(0)
    x_adv = Variable(x.data, requires_grad=True)
    for i in range(iteration):
        rec = recon(128, g, x_adv)
        rec = Variable(rec.data, requires_grad=True)
        ori_feature = net(rec).unsqueeze(0)
        cost = loss(ori_feature, targets).mean()
        if t:
            cost = -cost
        net.zero_grad()
        if rec.grad is not None:
            rec.grad.data.fill_(0)
        cost.backward()
        rec.grad.sign_()
        x_adv = x_adv.detach() + alpha * rec.grad.detach()
        x_adv = clamp(x_adv, x - eps, x + eps)
        x_adv = clamp(x_adv, lower_limit, upper_limit)
        x_adv = Variable(x_adv.data, requires_grad=True)
    return x_adv.data

def fgsm_face(net, x, targets, eps, alpha, iteration, t=False, random=True):
    loss = nn.CosineSimilarity()
    targets = net(targets).detach().unsqueeze(0)
    eta = torch.zeros_like(x).cuda()
    if random:
        for j in range(len(eps)):
            eta[:, j, :, :].uniform_(-eps[j][0][0].item(), eps[j][0][0].item())
    x_adv = clamp(x.data+eta, lower_limit, upper_limit)
    x_adv = Variable(x_adv, requires_grad=True)
    for i in range(iteration):
        outputs = net(x_adv).unsqueeze(0)
        cost = loss(outputs, targets).mean()
        if t:
            cost = -cost
        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        grad = x_adv.grad.sign()
        x_adv = x_adv.detach() + alpha * grad.detach()
        x_adv = clamp(x_adv, x-eps, x+eps)
        x_adv = clamp(x_adv, lower_limit, upper_limit)
        x_adv = Variable(x_adv.data, requires_grad=True)
    return x_adv.data

import matplotlib.pyplot as plt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def draw_roc(y_test, y_score, name):
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('fig/roc_%s.jpg' % name)
    return threshold

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
