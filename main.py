import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from PIL import Image

import os
import argparse
import numpy as np
import datetime
import json
import collections
import pathlib
import copy
from tqdm import tqdm

from resnet import *
from utils import *
from dataload import CIFAR10, CIFAR100


parser = argparse.ArgumentParser(description='Noisy Student CIFAR10/CIFAR100 ResNet')
parser.add_argument('--lr', default=0.128, type=float, help='learning rate')
parser.add_argument('--epochs', default=350, type=int, help='Total number of epochs')
parser.add_argument('--warm_up', default=10, type=int, help='number of epochs before main training starts')

parser.add_argument('--dataset', default='CIFAR100', type=str, help='Dataset [CIFAR10, CIFAR100]')
parser.add_argument('--outdir', default='results/', type=str, help='Output directory')
parser.add_argument('--model', default='ResNet18', type=str, help='[ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]')
parser.add_argument('--batch_size', default=256, type=int, help='Training batch size.')
parser.add_argument('--ts_iteration', default=3, type=int, help='number of student to teacher switch iterations')
parser.add_argument('--gradual_growth', default=True, type=bool, help='whether to increase student model size gradually')

parser.add_argument('--name', default='noisy_student', type=str, help='Name of the experiment')

args = parser.parse_args()
print(args.__dict__)

args.outdir = args.outdir + args.name + '/'

outdir = pathlib.Path(args.outdir + '_'.join(s for s in [args.model, args.dataset, args.err_method, str(args.err_rate)]))
outdir.mkdir(exist_ok=True, parents=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'CIFAR10':
    trainset = CIFAR10(mode='train', transform=transform_train, args=args)
    exp_set = CIFAR10(mode='aug_exp', transform=None, args=args)
    testset = CIFAR10(mode='test', transform=transform_test, args=args)

elif args.dataset == 'CIFAR100':
    trainset = CIFAR100(mode='train', transform=transform_train, args=args)
    exp_set = CIFAR100(mode='aug_exp', transform=None, args=args)
    testset = CIFAR100(mode='test', transform=transform_test, args=args)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
exp_loader = torch.utils.data.DataLoader(exp_set, batch_size=args.batch_size*5, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size*2, shuffle=False, num_workers=2)

def create_model(model_name):
    print('==> Building model..')
    # all ResNets here have Stochastic Depth & Dropout applied (Noisy Student)
    if model_name == 'ResNet152':
        model = ResNet152_Dropout(num_classes=len(testloader.dataset.classes), in_channels=3)
    elif model_name == 'ResNet101':
        model = ResNet101_Dropout(num_classes=len(testloader.dataset.classes), in_channels=3)
    elif model_name == 'ResNet50':
        model = ResNet50_Dropout(num_classes=len(testloader.dataset.classes), in_channels=3)
    elif model_name == 'ResNet34':
        model = ResNet34_Dropout(num_classes=len(testloader.dataset.classes), in_channels=3)
    elif model_name == 'ResNet18':
        model = ResNet18_Dropout(num_classes=len(testloader.dataset.classes), in_channels=3)
    elif model_name == 'Base':
        model = BaseModel(num_classes=len(testloader.dataset.classes))
    return model

# change this part if you want student model architecture to grow gradually in size.
if args.gradual_growth:
    model_list = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
    batch_sizes = [args.batch_size, args.batch_size, args.batch_size, int(args.batch_size/2), int(args.batch_size/2)]

else:
    model_list = []
    batch_sizes = []
    for i in args.ts_iteration:
        model_list.append(args.model_name)
        batch_sizes.append(args.batch_size)

model_teacher = create_model(model_list[0])
model_student = create_model(model_list[1])

print(model_list[:args.ts_iteration+1])
start_date = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d")

model_teacher = model_teacher.to(device)
model_student = model_student.to(device)

if device == 'cuda':
    model_teacher = torch.nn.DataParallel(model_teacher)
    model_student = torch.nn.DataParallel(model_student)
    cudnn.benchmark = True

optimizer_teacher = optim.SGD(model_teacher.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True, dampening=0)
scheduler_teacher = torch.optim.lr_scheduler.StepLR(optimizer_teacher, step_size=5, gamma=0.97)

optimizer_student = optim.SGD(model_student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True, dampening=0)
scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=5, gamma=0.97)

criterion = nn.CrossEntropyLoss()

# Training
def warmup(epoch, model, trainloader):
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    trainloader = tqdm(trainloader)
    trainloader.set_description('[%s %04d/%04d]' % ('[warmup]', epoch, args.warm_up))

    metrics = Accumulator()

    for batch_idx, (inputs, input_noaug, target, noisy_target, is_noise, dataset_index) in enumerate(trainloader):
        inputs, noisy_target = inputs.to(device), noisy_target.to(device)
        optimizer_teacher.zero_grad()

        outputs1 = model(inputs)
        loss_1 = criterion(outputs1, noisy_target)
        loss_1.backward()
        optimizer_teacher.step()

        train_loss += loss_1.item()
        _, predicted = outputs1.max(1)
        total += noisy_target.size(0)
        correct += predicted.eq(noisy_target).sum().item()

        total_acc = correct / total
        metrics.add_dict({
            'loss': loss_1.item() * noisy_target.size(0),
            'acc': total_acc * noisy_target.size(0),
        })
        postfix = metrics / total
        postfix['lr'] = optimizer_teacher.param_groups[0]['lr']
        trainloader.set_postfix(postfix)

    total_loss = train_loss / total
    total_acc = correct / total

    metrics /= total
    postfix['lr'] = optimizer_teacher.param_groups[0]['lr']

    log = collections.OrderedDict({
        'epoch': epoch,
        'train':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log


# Training
def train_student(epoch, model_student, model_teacher, labeled_loader, unlabeled_loader):
    model_student.train()
    model_teacher.eval()

    train_loss = 0
    correct = 0
    total = 0
    labeled_loader = tqdm(labeled_loader)
    labeled_loader.set_description('[%s %04d/%04d]' % ('[train]', epoch, args.epochs))

    metrics = Accumulator()
    iter_u = iter(unlabeled_loader)

    for batch_idx, (inputs, inputs_noaug, target, noisy_target, is_noise, dataset_index) in enumerate(labeled_loader):
        try:
            inputs_u, inputs_noaug_u, target_u, noisy_target_u, is_noise_u, index_u = next(iter_u)
        except StopIteration:
            iter_u = iter(unlabeled_loader)
            inputs_u, inputs_noaug_u, target_u, noisy_target_u, is_noise_u, index_u = next(iter_u)

        inputs_u = inputs_u.to(device)
        inputs_noaug_u = inputs_noaug_u.to(device)

        with torch.no_grad():
            # prediction of unlabeled data on teacher model (no augmentation)
            pseudo_logit = model_teacher(inputs_noaug_u)
            pseudo_label = F.softmax(pseudo_logit, dim=1).detach()
        
        outputs0 = model_student(inputs_u)
        outputs0 = F.log_softmax(outputs0, dim=1)

        loss_0 = F.kl_div(outputs0, pseudo_label, reduction='none')
        loss_0 = torch.sum(loss_0, dim=1)

        inputs, noisy_target = inputs.to(device), noisy_target.to(device)
        optimizer_student.zero_grad()

        outputs1 = model_student(inputs)
        loss_1 = criterion(outputs1, noisy_target)
        
        loss = torch.mean(loss_0) + loss_1
        
        loss.backward()
        optimizer_student.step()

        train_loss += loss.item()
        _, predicted = outputs1.max(1)
        total += noisy_target.size(0)
        correct += predicted.eq(noisy_target).sum().item()

        total_acc = correct / total
        metrics.add_dict({
            'loss': loss.item() * noisy_target.size(0),
            'acc': total_acc * noisy_target.size(0),
        })
        postfix = metrics / total
        postfix['lr'] = optimizer_student.param_groups[0]['lr']
        labeled_loader.set_postfix(postfix)

    total_loss = train_loss / total
    total_acc = correct / total

    metrics /= total
    postfix['lr'] = optimizer_student.param_groups[0]['lr']

    log = collections.OrderedDict({
        'epoch': epoch,
        'train':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log


def test(epoch, model, testloader, total_epoch):
    global best_acc
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    testloader = tqdm(testloader)
    testloader.set_description('[%s %04d/%04d]' % ('[*test]', epoch, total_epoch))

    metrics = Accumulator()

    with torch.no_grad():
        for batch_idx, (inputs, target, data_index) in enumerate(testloader):
            inputs, target = inputs.to(device), target.to(device)
            outputs1 = model(inputs)
            loss1 = criterion(outputs1, target)

            test_loss += loss1.item()
            _, predicted = outputs1.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            total_acc = correct / total
            metrics.add_dict({
                'loss': loss1.item() * target.size(0),
                'acc': total_acc * target.size(0),
            })
            postfix = metrics / total
            testloader.set_postfix(postfix)
            metrics /= total

    total_loss = test_loss / total
    total_acc = correct / total
    log = collections.OrderedDict({
        'epoch': epoch,
        'test':
            collections.OrderedDict({
                'loss': total_loss,
                'accuracy': total_acc,
            }),
    })
    return log, total_acc


if __name__ == "__main__":
    exp_logs = []
    exp_info = collections.OrderedDict({
        'model': model_list,
        'type': 'default',
        'arguments': args.__dict__,
    })

    exp_log = exp_info.copy()
    exp_logs.append(exp_log)
    save_json_file_withname(outdir, args.name, exp_logs)

    # save & load model
    # reset_model = os.path.join("model_save", "reset_model_file")
    # os.makedirs(os.path.dirname(os.path.abspath(reset_model)), exist_ok=True)
    # torch.save(model_teacher.state_dict(), reset_model)
    # model_student.load_state_dict(torch.load(reset_model))

    print("initial teacher train / train with all data")
    for epoch in range(args.warm_up):
        train_log = warmup(epoch, model_teacher, trainloader)
        if epoch % 10 == 0:
            test_log = test(epoch, model_teacher, testloader, args.warm_up)
        scheduler_teacher.step()

        exp_log = train_log.copy()
        exp_log.update(test_log)
        exp_logs.append(exp_log)
        save_json_file_withname(outdir, args.name, exp_logs)

    del optimizer_teacher, scheduler_teacher

    # student train start
    for i in range(args.ts_iteration):
        print('\n[{}/{}] iterative training on student'.format(i+1, args.ts_iteration))

        for epoch in range(args.epochs):
            # teacher model helps student model 
            # by providing 'pseudo label on unlabeled data' to the student model
            train_log = train_student(epoch, model_student, model_teacher, labeled_loader, unlabeled_loader)
            if epoch % 10 ==0:
                test_log = test(epoch, model_student, testloader, args.epochs)
            scheduler_student.step()

            exp_log = train_log.copy()
            exp_log.update(test_log)
            exp_logs.append(exp_log)
            save_json_file_withname(outdir, args.name, exp_logs)
        
        if i != args.ts_iteration - 1:
            model_teacher = model_student

            # create new student model, optimizer, scheduler etc. / change batch sizes accordingly depending on GPU Memory
            args.batch_size = batch_sizes[i+2]
            model_student = create_model(model_list[i+2])
            model_student = model_student.to(device)
            if device == 'cuda':
                model_student = torch.nn.DataParallel(model_student) 

            optimizer_student = optim.SGD(model_student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True, dampening=0)
            scheduler_student = torch.optim.lr_scheduler.StepLR(optimizer_student, step_size=5, gamma=0.97)
