import json
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from collections import defaultdict

import os

def generate_noise_data(label: torch.Tensor, eta: int, method: str, dataset: str):
    if method == "remove":
        method = "pair"

    # code for returning noise_or_not has been fixed.

    if method == "pair":
        sta = torch.randint_like(label, 1, 101)
        ret = torch.empty_like(label)
        ret.data = label.clone()
        ret[sta < eta] += 1
        # noise_or_not = torch.where(
        #     sta < eta, torch.ones_like(label), torch.zeros_like(label)
        # )
        # num_noisy_data = (noise_or_not == 1).sum(dim=0)
        ret[ret > label.max()] = label.min()
        noise_or_not = torch.where(
                ret != label, torch.ones_like(label), torch.zeros_like(label)
        )
        return ret, noise_or_not

    elif method == "symmetry":
        cnt = ((label.max() - label.min()).float() / (float(eta) / 100)).int()
        sta = torch.randint_like(label, 0, cnt)
        ret = torch.empty_like(label)
        ret.data = label.clone()
        # noise_or_not = torch.where(
        #     sta < label.max(), torch.ones_like(label), torch.zeros_like(label)
        # )
        # num_noisy_data = (noise_or_not == 1).sum(dim=0)
        ret[sta < label.max()] = sta[sta < label.max()]
        noise_or_not = torch.where(
                ret != label, torch.ones_like(label), torch.zeros_like(label)
        )
        return ret, noise_or_not
    elif method == "asymmetry":
        if dataset == "MNIST":
            """
            1 <- 7
            2 -> 7
            3 -> 8
            5 <-> 6
            """
            rand = torch.randint_like(label, 1, 101)
            ret = torch.empty_like(label)
            ret.data = label.clone()

            ret = torch.where(
                label == 7, torch.where(rand > eta, ret, torch.ones_like(label)), ret
            )
            ret = torch.where(
                label == 2,
                torch.where(rand > eta, ret, torch.ones_like(label) * 7),
                ret,
            )
            ret = torch.where(
                label == 3,
                torch.where(rand > eta, ret, torch.ones_like(label) * 8),
                ret,
            )
            ret = torch.where(
                label == 5,
                torch.where(rand > eta, ret, torch.ones_like(label) * 6),
                ret,
            )
            ret = torch.where(
                label == 6,
                torch.where(rand > eta, ret, torch.ones_like(label) * 5),
                ret,
            )

            noise_or_not = torch.where(
                ret != label, torch.ones_like(label), torch.zeros_like(label)
            )

            return ret, noise_or_not

        elif dataset == "CIFAR10":
            """
            (Patrini)Making Deep Neural Networks Robust to Label Noise
            
            mistakes:
            (0~9)-(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
            automobile <- truck (1 <- 9)
            bird -> airplane (2 -> 0)
            cat <-> dog (3 <-> 5)
            deer -> horse (4 -> 7)
            """
            rand = torch.randint_like(label, 1, 101)
            ret = torch.empty_like(label)
            ret.data = label.clone()

            ret = torch.where(
                label == 9, torch.where(rand > eta, ret, torch.ones_like(label)), ret
            )
            ret = torch.where(
                label == 2, torch.where(rand > eta, ret, torch.zeros_like(label)), ret
            )
            ret = torch.where(
                label == 3,
                torch.where(rand > eta, ret, torch.ones_like(label) * 5),
                ret,
            )
            ret = torch.where(
                label == 5,
                torch.where(rand > eta, ret, torch.ones_like(label) * 3),
                ret,
            )
            ret = torch.where(
                label == 4,
                torch.where(rand > eta, ret, torch.ones_like(label) * 7),
                ret,
            )

            noise_or_not = torch.where(
                ret != label, torch.ones_like(label), torch.zeros_like(label)
            )

            return ret, noise_or_not

        elif dataset == "CIFAR100":
            """
            mistakes are made only within the same superclasses
            
            e.g. aquatic mammals : 'beaver, dolphin, otter, seal, whale' 
            beaver is only mistaken as 'dolphin, otter, seal, whale'
            """
            rand = torch.randint_like(label, 1, 101)
            ret = torch.empty_like(label)
            ret.data = label.clone()

            superclass = [
                [4, 30, 55, 72, 95],
                [1, 32, 67, 73, 91],
                [54, 62, 70, 82, 92],
                [9, 10, 16, 28, 61],
                [0, 51, 53, 57, 83],
                [22, 39, 40, 86, 87],
                [5, 20, 25, 84, 94],
                [6, 7, 14, 18, 24],
                [3, 42, 43, 88, 97],
                [12, 17, 37, 68, 76],
                [23, 33, 49, 60, 71],
                [15, 19, 21, 31, 38],
                [34, 63, 64, 66, 75],
                [26, 45, 77, 79, 99],
                [2, 11, 35, 46, 98],
                [27, 29, 44, 78, 93],
                [36, 50, 65, 74, 80],
                [47, 52, 56, 59, 96],
                [8, 13, 48, 58, 90],
                [41, 69, 81, 85, 89],
            ]

            for sub in superclass:
                ret = torch.where(
                    label == sub[0],
                    torch.where(rand > eta, ret, torch.ones_like(label) * sub[1]),
                    ret,
                )
                ret = torch.where(
                    label == sub[1],
                    torch.where(rand > eta, ret, torch.ones_like(label) * sub[2]),
                    ret,
                )
                ret = torch.where(
                    label == sub[2],
                    torch.where(rand > eta, ret, torch.ones_like(label) * sub[3]),
                    ret,
                )
                ret = torch.where(
                    label == sub[3],
                    torch.where(rand > eta, ret, torch.ones_like(label) * sub[4]),
                    ret,
                )
                ret = torch.where(
                    label == sub[4],
                    torch.where(rand > eta, ret, torch.ones_like(label) * sub[0]),
                    ret,
                )

            noise_or_not = torch.where(
                ret != label, torch.ones_like(label), torch.zeros_like(label)
            )
            # num_noisy_data = (noise_or_not == 1).sum(dim=0)

            return ret, noise_or_not

    else:
        assert 0


def get_val_split_sampler(dataset, val_size=0.2, random_seed=None, shuffle=True):
    """Get Random Sampler for spliting dataset

    Args:
        dataset (Dataset): Dataset to be splited
        test_size (float): test size ratio
        random_seed (int): random seed
        shuffle (bool): shuffle

    Returns:
        train_sampler (SubsetRandomSampler): train sampler
        test_sampler (SubsetRandomSampler): test sampler
    """

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]
    return torch.utils.data.SubsetRandomSampler(train_indices), torch.utils.data.SubsetRandomSampler(test_indices)


def save_json_file(path, data):
    with open(f'{path}/log.json', "w") as outfile:
        json.dump(data, outfile, indent=2)


def save_json_file_withname(path, name, data):
    with open(f'{path}/{name}log.json', "w") as outfile:
        json.dump(data, outfile, indent=2)


def get_binary_stats(
    model,
    data_loader,
    result_dir: str,
    epoch: int,
    device: torch.device = torch.device("cpu"),
):
    """Save stats for model.

    Args:
        model (nn.Module): model object.
        data_loader (Dataloader): data loader for stats.
        result_dir (str): root result directory.
        device (torch.device): device object.

    Returns:
        filepath (str): filepath saved
        df (pd.DataFrame): pandas data frame
    """
    columns = [
        "loss (noise)",
        "is_noise",
        "y_pred",
        "y_true",
        "y_noise",
    ]
    df = pd.DataFrame(columns=columns)

    if data_loader.sampler is not None:
        total = len(data_loader.sampler)
    else:
        total = len(data_loader)

    pbar = tqdm(total=total, leave=False)

    print("collecting data...")

    model.eval()

    for data, target, noise_target, noise_or_not, index in data_loader:
        data, noise_target = data.to(device), noise_target.to(device)
        noise_or_not = noise_or_not.to(device)

        with torch.no_grad():
            # y_pred is the output before softmax layer.
            y_pred = model(data, noise_target)

        loss_noise = F.cross_entropy(y_pred, noise_or_not, reduction="none")

        probs = F.softmax(y_pred, dim=-1)

        df = df.append(
            pd.DataFrame(
                dict(
                    zip(
                        columns,
                        [
                            loss_noise.cpu().numpy(),
                            noise_or_not.cpu().numpy(),
                            y_pred.sort(descending=True)[1][:, 0].cpu().numpy(),
                            target.cpu().numpy(),
                            noise_target.cpu().numpy(),
                        ],
                    )
                )
            ),
            ignore_index=True,
        )
        pbar.update(data.size(0))
    pbar.close()

    # disable data augmentation
    # data_loader.dataset.set_aug(0)

    print("save stats...")

    model_name = "binary_class_trained_with_small_large_loss"

    dataset_name = data_loader.dataset.__class__.__name__.upper()
    err_method = getattr(data_loader.dataset, "err_method", None)
    err_rate = getattr(data_loader.dataset, "err_rate", None)

    filename = "{}_{}_{}_{}_Ep{}.csv".format(
        model_name, dataset_name, err_method, err_rate, epoch
    )
    filepath = os.path.join(result_dir, "stats", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    return filepath, df


def get_stats(
    model,
    data_loader,
    result_dir: str,
    epoch: int,
    device: torch.device = torch.device("cpu"),
):
    """Save stats for model.

    Args:
        model (nn.Module): model object.
        data_loader (Dataloader): data loader for stats.
        result_dir (str): root result directory.
        device (torch.device): device object.

    Returns:
        filepath (str): filepath saved
        df (pd.DataFrame): pandas data frame
    """
    columns = [
        "index",
        "loss",
        "loss (noise)_0",
        "loss (noise)_1",
        "loss (noise)_2",
        "loss_ae",
        # "loss (noise)_3",
        # "loss (noise)_4",
        # "loss (noise)_5",
        # "loss (noise)_6",
        # "loss (noise)_7",
        # "loss (noise)_8",
        # "entropy",
        # "jsd",
        # "jsd (noise)",
        # "jsd_aug",
        # "mse_logit",
        "y_pred0",
        "y_pred1",
        "y_pred2",
        # "y_pred3",
        # "y_pred4",
        # "y_pred5",
        # "y_pred6",
        # "y_pred7",
        # "y_pred8",
        "y_pred1_2",
        # "y_pred7_8",
        "y_true",
        "y_noise",
        "is_noise",
    ]
    df = pd.DataFrame(columns=columns)

    if data_loader.sampler is not None:
        total = len(data_loader.sampler)
    else:
        total = len(data_loader)

    pbar = tqdm(total=total, leave=False)

    print("collecting data...")

    model.eval()

    criterion = nn.MSELoss(reduction='none')

    # for data0, data1, data2, data3, data4, data5, data6, data7, data8, target, noise_target, noise_or_not, index in data_loader:
    for data0, data1, data2, target, noise_target, noise_or_not, index in data_loader:
        data0, data1, data2, target = data0.to(device), data1.to(device), data2.to(device), target.to(device)
        # data3, data4, data5, data6, data7, data8 = data3.to(device), data4.to(device), data5.to(device), data6.to(device), data7.to(device), data8.to(device)
        noise_target = noise_target.to(device)

        with torch.no_grad():
            # y_pred is the output before softmax layer.
            y_pred0 = model(data0)
            y_pred1 = model(data1)
            y_pred2 = model(data2)
            # y_pred3 = model(data3)
            # y_pred4 = model(data4)
            # y_pred5 = model(data5)
            # y_pred6 = model(data6)
            # y_pred7 = model(data7)
            # y_pred8 = model(data8)
            y_pred1_2 = (y_pred1 + y_pred2)/2
            # y_pred7_8 = (y_pred7 + y_pred8)/2
        
        # loss_ae = criterion(output, data0)
        # loss_ae_sum = loss_ae.view(loss_ae.shape[0], -1).sum(1, keepdim=True)
        # print(loss_ae_sum.cpu().numpy().squeeze())

        # print(loss_ae_sum.cpu().numpy().squeeze().shape)

        loss = F.cross_entropy(y_pred1, target, reduction="none")
        loss_noise_0 = F.cross_entropy(y_pred0, noise_target, reduction="none")
        loss_noise_1 = F.cross_entropy(y_pred1, noise_target, reduction="none")
        loss_noise_2 = F.cross_entropy(y_pred2, noise_target, reduction="none")

        # loss_noise_3 = F.cross_entropy(y_pred3, noise_target, reduction="none")
        # loss_noise_4 = F.cross_entropy(y_pred4, noise_target, reduction="none")
        # loss_noise_5 = F.cross_entropy(y_pred5, noise_target, reduction="none")
        # loss_noise_6 = F.cross_entropy(y_pred6, noise_target, reduction="none")
        # loss_noise_7 = F.cross_entropy(y_pred7, noise_target, reduction="none")
        # loss_noise_8 = F.cross_entropy(y_pred8, noise_target, reduction="none")

        # probs = F.softmax(y_pred1, dim=-1)
        # probs2 = F.softmax(y_pred2, dim=-1)
        
        # entropy = -torch.sum(probs.log()*probs, dim=1)

        # T = 0.1 # temperature for sharpening
        # probs = probs**(1/T) # sharpening / probs is (N, D)-shaped probability distribution
        # probs2 = probs2**(1/T)
        
        # jsd = jansen_shannon_dist(target, probs)
        # jsd_noise = jansen_shannon_dist(noise_target, probs)
        # jsd_aug = jansen_shannon_dist(probs, probs2)

        # y_pred1 = y_pred1**(1/T)
        # y_pred2 = y_pred2**(1/T)

        # mse_logit = logit_MSE(y_pred1, y_pred2)

        df = df.append(
            pd.DataFrame(
                dict(
                    zip(
                        columns,
                        [
                            index.numpy(),
                            loss.cpu().numpy(),
                            loss_noise_0.cpu().numpy(),
                            loss_noise_1.cpu().numpy(),
                            loss_noise_2.cpu().numpy(),
                            # loss_ae_sum.cpu().numpy().squeeze(),
                            # loss_noise_3.cpu().numpy(),
                            # loss_noise_4.cpu().numpy(),
                            # loss_noise_5.cpu().numpy(),
                            # loss_noise_6.cpu().numpy(),
                            # loss_noise_7.cpu().numpy(),
                            # loss_noise_8.cpu().numpy(),
                            # entropy.cpu().numpy(),
                            # jsd.cpu().numpy(),
                            # jsd_noise.cpu().numpy(),
                            # jsd_aug.cpu().numpy(),
                            # mse_logit.cpu().numpy(),
                            y_pred0.sort(descending=True)[1][:, 0].cpu().numpy(),
                            y_pred1.sort(descending=True)[1][:, 0].cpu().numpy(),
                            y_pred2.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred3.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred4.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred5.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred6.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred7.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred8.sort(descending=True)[1][:, 0].cpu().numpy(),
                            y_pred1_2.sort(descending=True)[1][:, 0].cpu().numpy(),
                            # y_pred7_8.sort(descending=True)[1][:, 0].cpu().numpy(),
                            target.cpu().numpy(),
                            noise_target.cpu().numpy(),
                            noise_or_not.cpu().numpy(),
                        ],
                    )
                )
            ),
            ignore_index=True,
        )
        pbar.update(data1.size(0))
    pbar.close()

    # disable data augmentation
    # data_loader.dataset.set_aug(0)

    print("save stats...")

    if hasattr(model, "module"):
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__

    dataset_name = data_loader.dataset.__class__.__name__.upper()
    err_method = getattr(data_loader.dataset, "err_method", None)
    err_rate = getattr(data_loader.dataset, "err_rate", None)

    filename = "{}_{}_{}_{}_Ep{}.csv".format(
        model_name, dataset_name, err_method, err_rate, epoch
    )
    filepath = os.path.join(result_dir, "stats", filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

    return


def jansen_shannon_dist(y_true, y_pred):
    if y_true.size() != y_pred.size():
        # when y_true has the index of the class
        y_true = torch.nn.functional.one_hot(y_true, num_classes=y_pred.size(1)).float()
    dist_mean = (y_true + y_pred) * 0.5

    # y_true log(1+p)로 계산 안할경우 nan 값 나옴
    # kldiv1 = torch.sum(y_true * (torch.log1p(y_true) - torch.log1p(dist_mean)), dim=1)
    
    kldiv1 = torch.sum(F.kl_div(F.log_softmax(y_true, -1), dist_mean, reduction='none'), dim=1)
    kldiv2 = torch.sum(F.kl_div(F.log_softmax(y_pred, -1), dist_mean, reduction='none'), dim=1)

    jsd = 0.5 * (kldiv1 + kldiv2)
    return jsd


def logit_MSE(logit1, logit2):
    return torch.sum(F.mse_loss(logit1, logit2, reduction='none'), dim=1)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        filename (str): Checkpoint filename
                        Default: checkpoint.pt
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
    """

    def __init__(self, filename="checkpoint.pt", patience=7, verbose=False, delta=0):
        self.filename = filename
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

        self.filepath = os.path.abspath(filename)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def __call__(self, val_loss, ):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return False

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                "Validation loss decreased ({:.3f} --> {:.3f}).  Saving model ...".format(
                    self.val_loss_min, val_loss
                )
            )
        torch.save(model.state_dict(), self.filepath)
        self.val_loss_min = val_loss


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone