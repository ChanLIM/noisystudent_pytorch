# Noisy Student pytorch

This is implementation of Noisy Student [[paper]](https://arxiv.org/abs/1911.04252) [[tensorflow github code]](https://github.com/google-research/noisystudent) in PyTorch using smaller dataset(CIFAR10/CIFAR100) and smaller model architecture(ResNet).

Original paper uses ImageNet2012, on top of JFT dataset as external dataset to push up the classification performance.

However, this code aims on reproducing the main idea on PyTorch.

The code will be using CIFAR100 instead of ImageNet2012, on top of CIFAR10 instead of JFT, as the external dataset.

# TO DO

## Noising Student
### Input Noise
- RandAugment : Substituted to AutoAugment
### Model Noise
- Dropout : O
- Stochastic Depth : X

## LR Decay
- gamma 0.97 (every 4.8 epochs if small model 700 epochs / every 2.4 epochs if large model 350 epochs) : O? X?

## Train-Test Resolution Discrepancy
- increase test time crop size & fine tune : X

## Other Techniques
- filter images that the teacher has low confidences on : O
- balance the number of unlabeled images for each class : X


To run the code,

            python main.py \
                --lr=0.001 \
                --dataset='CIFAR10'
