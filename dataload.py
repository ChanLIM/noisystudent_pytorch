import os
import numpy as np

import torch
from torch.utils import data

import torchvision
import torchvision.transforms as transforms
from PIL import Image

from archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from augmentations import *


class CutoutDefault(object):
	"""
	Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
	"""
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		if self.length <= 0:
			return img
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img


class Augmentation(object):
	def __init__(self, policies):
		self.policies = policies

	def __call__(self, img):
		for _ in range(1):
			policy = random.choice(self.policies)
			for name, pr, level in policy:
				if random.random() > pr:
					continue
				img = apply_augment(img, name, level)
		return img


class supervised_Dataset(data.Dataset):
	def __init__(self, dataset):
		self.data = dataset.data
		self.targets = dataset.targets
		self.noisy_target = dataset.noisy_target
		self.is_noise = dataset.is_noise
		self.transform_default = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	def __getitem__(self, index):
		data, target = self.data[index], self.targets[index]
		data = self.transform_default(Image.fromarray(data, 'RGB'))
		return data, target, self.noisy_target[index], self.is_noise[index], index

	def __len__(self):
		return len(self.data)


class unsupervised_Dataset(data.Dataset):
	def __init__(self, dataset):
		self.data = dataset.data
		self.targets = dataset.targets
		self.noisy_target = dataset.noisy_target
		self.is_noise = dataset.is_noise

		self.transform_default = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		self.transform_aug = transforms.Compose([
			# flips with given probability. if p=1, always flip.
			transforms.RandomHorizontalFlip(p=1),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	def __getitem__(self, index):
		data, target = self.data[index], self.targets[index]

		data_orig = self.transform_default(Image.fromarray(data, 'RGB'))
		data_aug = self.transform_aug(Image.fromarray(data, 'RGB'))

		return data_orig, data_aug, target, self.noisy_target[index], self.is_noise[index], index

	def __len__(self):
		return len(self.data)


class CIFAR10(torchvision.datasets.CIFAR10):
	def __init__(self, mode, args):
		train = True if mode in ['train', 'unlabeled'] else False
		super().__init__(root='./data/CIFAR10', train=train, download=True, transform=None)
		self.mode = mode
		self.dataname = "CIFAR10"
		self.classes = np.arange(10)

	def __getitem__(self, index):
		data, target = self.data[index], self.targets[index]

		if self.mode == "train":
			self.set_aug(5)
			data0 = self.transform(Image.fromarray(data, 'RGB'))
			self.set_aug(0)
			data_noaug = self.transform(Image.fromarray(data, 'RGB'))
			return data0, data_noaug, target, index

		elif self.mode == "test":
			self.set_aug(0)
			data = self.transform(Image.fromarray(data, 'RGB'))
			return data, target, index

	def set_aug(self, method):
		if method == 0:
			# print("Using No Transformations")
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

		elif method == 1:
			# print("Using 100% Crop")
			self.transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif method == 2:
			# print("Using 100% HorizontalFlips")
			self.transform = transforms.Compose([
				# flips with given probability. if p=1, always flip.
				transforms.RandomHorizontalFlip(p=1),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif method == 3: # most widely used data augmentation for CIFAR dataset
			# print("Using Random Crop & Random Horizontal Flip")
			self.transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4), # always crop
				transforms.RandomHorizontalFlip(), # flips with 0.5 probability
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

		elif method == 5:
			# AutoAugment & CutOut
			transform_train = transforms.Compose([
				transforms.RandomCrop(32, padding=4), # always crop
				transforms.RandomHorizontalFlip(), # flips with 0.5 probability
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			autoaug = transforms.Compose([])
			autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
			transform_train.transforms.insert(0, autoaug)
			transform_train.transforms.append(CutoutDefault(16))
			self.transform = transform_train


class CIFAR100(torchvision.datasets.CIFAR100):
	def __init__(self, mode, args):
		train = True if mode =='train' else False
		super().__init__(root='./data/CIFAR100', train=train, download=True, transform=None)
		self.mode = mode
		self.dataname = "CIFAR100"
		self.classes = np.arange(100)

	def __getitem__(self, index):
		data, target = self.data[index], self.targets[index]
		if self.mode == "train":
			self.set_aug(5)
			data0 = self.transform(Image.fromarray(data, 'RGB'))
			self.set_aug(0)
			data_noaug = self.transform(Image.fromarray(data, 'RGB'))
			return data0, data_noaug, target, index
				
		elif self.mode == "test":
			self.set_aug(0)
			data = self.transform(Image.fromarray(data, 'RGB'))
			return data, target, index

	def set_aug(self, method):
		'''
		Set Augmentation Options
		0: No augmentation
		1: 100% crop
		2: 100% horizontal flip
		3: 100% crop & 50% horizontal flip (commonly used augmentation)
		4: Augmentation policy found by AutoAugment
		'''
		if method == 0:
			# print("Using No Transformations")
			self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

		elif method == 1:
			# print("Using 100% Crop")
			self.transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif method == 2:
			# print("Using 100% HorizontalFlips")
			self.transform = transforms.Compose([
				# flips with given probability. if p=1, always flip.
				transforms.RandomHorizontalFlip(p=1),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif method == 3: # most widely used data augmentation for CIFAR dataset
			# print("Using Random Crop & Random Horizontal Flip")
			self.transform = transforms.Compose([
				transforms.RandomCrop(32, padding=4), # always crop
				transforms.RandomHorizontalFlip(), # flips with 0.5 probability
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		elif method == 5:
			transform_train = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			autoaug = transforms.Compose([])
			autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
			transform_train.transforms.insert(0, autoaug)
			transform_train.transforms.append(CutoutDefault(16))
			self.transform = transform_train
