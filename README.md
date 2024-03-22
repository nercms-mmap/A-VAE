# Manifold Projection for Adversarial Defense on Face Recognition
This repo is the official implementation of ["Manifold Projection for Adversarial Defense on Face Recognition"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750290.pdf) by Jianli Zhou, Chao Liang and Jun Chen.
## Abstract
Although deep convolutional neural network based face recognition system has achieved remarkable success, it is susceptible to adversarial images: carefully constructed imperceptible perturbations can easily mislead deep neural networks. A recent study has shown that in addition to regular off-manifold adversarial images, there are also adversarial images on the manifold. In this paper, we propose Adversarial Variational AutoEncoder (A-VAE), a novel framework to tackle both types of attacks. We hypothesize that both off-manifold and on-manifold attacks move the image away from the high probability region of image manifold. We utilize variational autoencoder (VAE) to estimate the lower bound of the log-likelihood of image and explore to project the input images back into the high probability regions of image manifold again. At inference time, our model synthesizes multiple similar realizations of a given image by random sampling, then the nearest neighbor of the given image is selected as the final input of the face recognition model. As a preprocessing operation, our method is attack-agnostic and can adapt to a wide range of resolutions. The experimental results on LFW demonstrate that our method achieves state-of-the-art defense success rate against conventional off-manifold attacks such as FGSM, PGD, and C&W under both grey-box and white-box settings, and even on-manifold attack.
## Approach
![图片名称](https://github.com/chenzhan1/test/blob/main/assets/framework.png)
## Training
We have provided the [CASIA-Webface](https://pan.baidu.com/s/1oSfvUlB35cWi--LIgL8-zQ?pwd=gim5) dataset for model training on Baiduyun. Please unzip it and put it into the `./dataset/` folder.
You can train A-VAE on CASIA-Webface with default settings using the following code:
```
cd avae_train
python main.py
```
Of course, you can also directly use the pre-trained [A-VAE](https://pan.baidu.com/s/1cSmNLyJL33x888GpdzMpnw?pwd=cfqg) model for testing, where the number of model iterations is 144002.
## Test
We have provided the [LFW](https://pan.baidu.com/s/1oSfvUlB35cWi--LIgL8-zQ?pwd=gim5) dataset for model test on Baiduyun. Please unzip it and put it into the `./dataset/` folder. In the meantime, we use the pre-trained [ResNet-50](https://pan.baidu.com/s/1Tiz3lki2tpCm5ktJALvYpQ?pwd=o723) as the face classifier during the testing process.
You can modify the following lines according to your needs to adjust the testing content, such as whether to use A-VAE, adopt gray-box or white-box attacks, and which attack method to use, etc.
```
avae_defense = False #['True','False']
```
```
white_box = False #['True','False']
```
```
if not white_box:
    # c&w attack
    inputs = cw_ut(net, inputs, targets, to_numpy=False)
    # fgsm/pgd attack
    # inputs = fgsm_face(net, inputs, targets, epsilon, alpha, iteration, t=True, random=True)
else:
    inputs = fgsm_w(net, g_running, inputs, targets, epsilon, alpha, iteration, t=True)
```

Then, You can test A-VAE on LFW using the following code:
```
cd avae_test
python eval.py
```
## Citation
If you find this repository useful, please consider giving ⭐ or citing:
```
@inproceedings{zhou2020manifold,
  title={Manifold projection for adversarial defense on face recognition},
  author={Zhou, Jianli and Liang, Chao and Chen, Jun},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XXX 16},
  pages={288--305},
  year={2020},
  organization={Springer}
}
```
