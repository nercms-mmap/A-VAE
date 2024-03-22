import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from torchvision import transforms
import cw
from utils import get_image, fgsm_face, fgsm_w, save_img,std, loadModel, on_attack, recon, load_state_dict
from model13 import StyledGenerator
from models import *



def veri(t, eps, iteration):
    r1, r2 = 0, 0
    j = 0
    m = 0
    epsilon = eps * (1 / 255.) / std
    if iteration == 1:
        alpha = epsilon
    else:
        alpha = (4 / 255.) / std
    batch_size = 10
    pairs = open(pairs_r, 'r')
    lines = pairs.readlines()[1:]
    cls = 'lfw_clean'
    inputs = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32).cuda()
    targets = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32).cuda()

    # c&w attack
    cw_ut = cw.L2Adversary(targeted=False,
                               confidence=0.0,
                               search_steps=10,
                               optimizer_lr=10)

    cw_t = cw.L2Adversary(targeted=True,
                               confidence=0.0,
                               search_steps=10,
                               optimizer_lr=10)

    for n in range(1):
        for i in range(300): #[117, 119, 178, 132]: #range(300):
            line = lines[n * 600 + i]
            name, n1, n2 = line.split('\t')
            n1, n2 = int(n1), int(n2)
            input = get_image(os.path.join(dataset_dir, name), '%s_%04d.jpg' % (name, n1), transform)
            target = get_image(os.path.join(dataset_dir, name), '%s_%04d.jpg' % (name, n2), transform)
            inputs[j] = input
            targets[j] = target
            j += 1
            if j % batch_size == 0:
                j = 0
                if on_manifold_attack:
                    inputs, _ = oa_ag.apply(net, inputs, targets, epsilon, t, iteration, t=True)
                elif off_manifold_attack:
                    if not white_box:
                        # c&w attack
                        inputs = cw_ut(net, inputs, targets, to_numpy=False)
                        # fgsm/pgd attack
                        # inputs = fgsm_face(net, inputs, targets, epsilon, alpha, iteration, t=True, random=True)
                    else:
                        inputs = fgsm_w(net, g_running, inputs, targets, epsilon, alpha, iteration, t=True)
                # save adversarial image
                # for k in range(batch_size):
                #     save_img(save_dir, m, inputs[k])
                #     m+=1
                if avae_defense:
                    inputs = recon(128, g_running, inputs.cuda())
                fea1 = net(inputs)
                fea2 = net(targets)
                cosS = CosSimi(fea1, fea2).data.cpu().numpy()
                r1 += np.sum(cosS >= t)

        for i in range(300):
            line = lines[n * 600 + i + 300]
            name1, n1, name2, n2 = line.split('\t')
            n1, n2 = int(n1), int(n2)
            input = get_image(os.path.join(dataset_dir, name1), '%s_%04d.jpg' % (name1, n1), transform)
            target = get_image(os.path.join(dataset_dir, name2), '%s_%04d.jpg' % (name2, n2), transform)
            inputs[j] = input
            targets[j] = target
            j += 1
            if j % batch_size == 0:
                j = 0
                if on_manifold_attack:
                    inputs, _ = oa_ag.apply(net, inputs, targets, epsilon, t, iteration, t=False)
                elif off_manifold_attack:
                    if not white_box:
                        # c&w attack
                        inputs = cw_t(net, inputs, targets, to_numpy=False)
                        # fgsm/pgd attack
                        # inputs = fgsm_face(net, inputs, targets, epsilon, alpha, iteration, t=False, random=True)
                    else:
                        inputs = fgsm_w(net, g_running, inputs, targets, epsilon, alpha, iteration, t=False)
                # for k in range(batch_size):
                #     save_img(save_dir, m, inputs[k])
                #     m+=1
                if avae_defense:
                    inputs = recon(128, g_running, inputs.cuda())
                fea1 = net(inputs)
                fea2 = net(targets)
                cosS = CosSimi(fea1, fea2).data.cpu().numpy()
                r2 += np.sum(cosS < t)

    p1 = round(float(r1) / (300 * (n+1)), 4)
    p2 = round(float(r2) / (300 * (n+1)), 4)
    return p1, p2, (p1+p2)/2


if __name__ == '__main__':
    pairs_r = './pairs.txt'
    dataset_dir = '../dataset/lfw_crop_128' # test dataset
    model_dir = './resnet50_ft_weight.pkl'
    generator_dir = './144002.model'        # A-VAE model
    save_dir = './img/'
    avae_defense = False
    off_manifold_attack = False
    on_manifold_attack = False
    white_box = False

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.0039215, 0.0039215, 0.0039215))
    ])

    # transform = transforms.Compose([
    #     transforms.Resize(128),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # net = CBAMResNet(50, feature_dim=512, mode='ir')
    # net.cuda()
    # checkpoint = torch.load(model_dir)['net_state_dict']  # 1. Iter_087000_net.ckpt 2. Iter_246000_net.ckpt
    # net, device = loadModel(net, checkpoint)

    net = resnet50()
    load_state_dict(net, model_dir)
    net.eval()
    net = net.cuda()

    CosSimi = torch.nn.CosineSimilarity()

    if avae_defense or on_manifold_attack:
        g_running = StyledGenerator(512).cuda()
        g_running.eval()
        g_checkpoint = torch.load(generator_dir)
        g_running.load_state_dict(g_checkpoint)
        if on_manifold_attack:
            oa_ag = on_attack(g_running)

    thre = 0.41

    prec1, prec2, mean = veri(thre, 0, 0)   # clean
    record = '%.4f, %.4f., %.4f. \n' % (prec1, prec2, mean)
    print(record)

    # config = [(4, 1), (8, 1), (8, 8)]  # pgd
    # config = [(16, 10)]  # on_manifold
    # with open('record.txt', 'a') as f:
    #     ss = []
    #     for (eps, iter) in config:
    #         prec1, prec2, mean = veri(thre, eps, iter)
    #         record = 'eps:%d, iter:%d, %.4f, %.4f., %.4f. \n' % (eps, iter, prec1, prec2, mean)
    #         print(record)
