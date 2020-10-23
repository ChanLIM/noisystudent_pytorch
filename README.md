# Noisy Student pytorch

This is implementation of Noisy Student [[paper]](https://arxiv.org/abs/1911.04252)[[tensorflow]](https://github.com/google-research/noisystudent) in PyTorch using smaller dataset(CIFAR10/CIFAR100) and smaller model architecture(ResNet).

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
