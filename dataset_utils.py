import random
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import math
import functools

from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom
from data_aug_utils import AutoAugment

# 新增 medmnist 导入
import medmnist
from medmnist import DermaMNIST

__all__ = ['partition_data', 'get_dataloader']


# ----------------------------------------------------------------------
# 原有数据集加载函数（保持不变）
# ----------------------------------------------------------------------
def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    xray_train_ds = ImageFolder_custom(datadir + '/train/', transform=None)
    xray_test_ds = ImageFolder_custom(datadir + '/val/', transform=None)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


# ----------------------------------------------------------------------
# 新增 DermaMNIST 加载函数
# ----------------------------------------------------------------------
def load_dermamnist_data(datadir):
    """
    加载 DermaMNIST 数据集。
    注意：medmnist 返回的 img 已经是 PIL.Image，因此 transform 中不能使用 ToPILImage()。
    将图像上采样到 64x64 以提升分辨率（原始为 28x28）。
    """
    BANDWIDTH = 64   # 上采样到 64x64
    train_transform = transforms.Compose([
        transforms.Resize((BANDWIDTH, BANDWIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((BANDWIDTH, BANDWIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DermaMNIST(split='train', transform=train_transform, download=True, root=datadir)
    test_dataset = DermaMNIST(split='test', transform=test_transform, download=True, root=datadir)

    X_train = train_dataset.imgs   # shape (N, 28, 28, 3)
    y_train = train_dataset.labels  # shape (N, 1)
    X_test = test_dataset.imgs
    y_test = test_dataset.labels
    return (X_train, y_train, X_test, y_test)


# ----------------------------------------------------------------------
# 狄利克雷分配辅助函数（从第二段代码完整复制）
# ----------------------------------------------------------------------
def build_non_iid_by_dirichlet(seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    """完全保留第二个代码的狄利克雷分配逻辑，无任何修改"""
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10  # 辅助节点数（优化大客户端数量场景）
    assert n_auxi_workers <= n_workers

    # 1. 随机打乱样本-标签映射
    random_state.shuffle(indices2targets)

    # 2. 分割样本集（适配大客户端数量）
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers if idx < num_splits - 1 
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    
    for idx, _n_workers in enumerate(split_n_workers):
        to_index = from_index + int(_n_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[from_index: (num_indices if idx == num_splits - 1 else to_index)]
        )
        from_index = to_index

    # 3. 为每个分割集分配客户端
    idx_batch = []
    remaining_workers = n_workers
    for _targets in splitted_targets:
        _targets = np.array(_targets)
        _targets_size = len(_targets)
        _n_workers = min(n_auxi_workers, remaining_workers)
        remaining_workers -= _n_workers

        min_size = 0
        _idx_batch = None
        # 循环确保每个客户端至少有基础样本数
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            
            # 按类别分配样本
            for _class in range(num_classes):
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]  # 获取当前类别的所有样本索引
                
                if len(idx_class) == 0:
                    continue  # 跳过无样本的类别
                
                # 狄利克雷采样分配比例
                proportions = random_state.dirichlet(np.repeat(non_iid_alpha, _n_workers))
                # 平衡客户端样本数（避免某客户端样本过多）
                proportions = np.array([
                    p * (len(idx_j) < _targets_size / _n_workers)
                    for p, idx_j in zip(proportions, _idx_batch)
                ])
                proportions = proportions / proportions.sum()  # 重归一化比例
                
                # 分割样本并分配给客户端
                split_points = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                _idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(_idx_batch, np.split(idx_class, split_points))
                ]
            
            min_size = min([len(idx_j) for idx_j in _idx_batch])  # 更新最小样本数
        
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch


def partition_balance(idxs, num_split):
    """完全保留第二个代码的均衡分割逻辑，无任何修改"""
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])  # 前r个分区多1个样本
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part
    
    return parts


def get_client_class_distribution(list_client2indices, list_label2indices, num_classes):
    """完全保留第二个代码的分布分析逻辑，适配第一个代码的字典格式输入"""
    # 适配：list_label2indices是第一个代码的字典{类别: 索引数组}，转为集合映射
    label_idx_map = {label: set(indices) for label, indices in list_label2indices.items()}
    client_class_dist = []
    client_vacant_classes = []
    
    for indices in list_client2indices:
        class_counts = {}
        for idx in indices:
            for label, idx_set in label_idx_map.items():
                if idx in idx_set:
                    class_counts[label] = class_counts.get(label, 0) + 1
                    break
        client_class_dist.append(class_counts)
        vacant_classes = [label for label in range(num_classes) if label not in class_counts]
        client_vacant_classes.append(vacant_classes)
    
    return client_class_dist, client_vacant_classes


# ----------------------------------------------------------------------
# 主分区函数（已添加 DermaMNIST 支持）
# ----------------------------------------------------------------------
def partition_data(dataset, datadir, partition, n_parties, alpha=0.4, class_per_client=2, balance=False):
    # 1. 加载数据集（新增 dermamnist 分支）
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'dermamnist':
        X_train, y_train, X_test, y_test = load_dermamnist_data(datadir)
    else:
        raise NotImplementedError("dataset not imeplemented")  # 保留原始拼写

    n_train = y_train.shape[0]
    party2dataidx = {}

    # 2. IID 分区
    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        party2dataidx = {i: batch_idxs[i] for i in range(n_parties)}

    # 3. noniid-labeldir 分区（新增 dermamnist 类别数）
    elif partition == "noniid-labeldir":
        min_size = 0
        party2dataidx = {}
        least_samples = 10
        min_require_size = 10
        num_classes = 10
        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'tinyimagenet':
            num_classes = 200
        elif dataset == 'dermamnist':
            num_classes = 7
        class_per_client = num_classes * 0.2
        idxs = np.array(range(len(y_train)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[y_train == i])

        class_num_per_client = [class_per_client for _ in range(n_parties)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(n_parties):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(n_parties/num_classes*class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                if dataset == 'cifar10':
                    num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_all_samples, num_selected_clients-1).tolist()
                else:
                    num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in party2dataidx.keys():
                    party2dataidx[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    party2dataidx[client] = np.append(party2dataidx[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    # 4. noniid 分区（新增 dermamnist 长尾跳过逻辑）
    elif partition == "noniid":
        imb_factor = 0.01
        imb_type = 'exp'
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'dermamnist':
            K = 7

        party2dataidx = {}
        list_label2indices_train = {k: np.where(y_train == k)[0] for k in range(K)}

        # 长尾压缩：仅对非 dermamnist 数据集执行
        if dataset != 'dermamnist':
            def _get_img_num_per_cls(list_label2indices, num_classes, imb_factor, imb_type):
                img_max = len(list_label2indices[0])
                img_num_per_cls = []
                if imb_type == 'exp':
                    for cls_idx in range(num_classes):
                        img_num = int(img_max * (imb_factor ** (cls_idx / (num_classes - 1.0))))
                        img_num_per_cls.append(img_num)
                elif imb_type == 'step':
                    num_step = 4
                    per_step = num_classes // num_step
                    img_num_per_cls = []
                    for s in range(num_step):
                        for _ in range(per_step):
                            img_num = int(img_max * (imb_factor ** s))
                            img_num_per_cls.append(img_num)
                    for _ in range(num_classes % num_step):
                        img_num = int(img_max * (imb_factor ** (num_step - 1)))
                        img_num_per_cls.append(img_num)
                else:
                    raise ValueError("Invalid imb_type, expected 'exp' or 'step'")
                return img_num_per_cls

            img_num_list = _get_img_num_per_cls(list_label2indices_train, K, imb_factor, imb_type)
            print('img_num_class:', img_num_list)

            long_tail_indices = []
            for cls_idx, img_num in enumerate(img_num_list):
                indices = list_label2indices_train[cls_idx]
                np.random.shuffle(indices)
                long_tail_indices.extend(indices[:img_num])

            long_tail_indices = np.array(long_tail_indices)
            np.random.shuffle(long_tail_indices)
            long_tail_y = y_train[long_tail_indices]
        else:
            # DermaMNIST：直接使用全部训练样本，保留原始长尾分布
            print(f"[{dataset}] 原始类别样本数：")
            for k in range(K):
                print(f"  类别 {k}: {len(list_label2indices_train[k])} 张")
            print(f"[{dataset}] 跳过全局长尾压缩，使用全部 {n_train} 个训练样本。")
            long_tail_indices = np.arange(n_train)
            np.random.shuffle(long_tail_indices)
            long_tail_y = y_train[long_tail_indices]

        total_samples = long_tail_indices.shape[0]

        # 构建狄利克雷分配所需的 (样本索引, 标签) 列表
        indices2targets = []
        for label in range(K):
            label_idx_in_longtail = np.where(long_tail_y == label)[0]
            label_indices = long_tail_indices[label_idx_in_longtail].tolist()
            for idx in label_indices:
                indices2targets.append((idx, label))

        print(f"[{dataset}] 使用狄利克雷分布分配数据（alpha={alpha}, 客户端数={n_parties}）")
        batch_indices = build_non_iid_by_dirichlet(
            seed=42,
            indices2targets=indices2targets,
            non_iid_alpha=alpha,
            num_classes=K,
            num_indices=len(indices2targets),
            n_workers=n_parties
        )

        indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
        list_client2indices = partition_balance(indices_dirichlet, n_parties)

        client_sample_counts = [len(indices) for indices in list_client2indices]
        while min(client_sample_counts) < min_require_size:
            np.random.shuffle(indices_dirichlet)
            list_client2indices = partition_balance(indices_dirichlet, n_parties)
            client_sample_counts = [len(indices) for indices in list_client2indices]

        for j in range(n_parties):
            np.random.shuffle(list_client2indices[j])
            party2dataidx[j] = np.array(list_client2indices[j])

        # 打印分配摘要
        client_class_dist, client_vacant_classes = get_client_class_distribution(
            list_client2indices, list_label2indices_train, K
        )
        print(f"\n[{dataset}] 客户端数据分配摘要:")
        print(f"客户端数量: {n_parties}")
        print(f"每个客户端样本数: {client_sample_counts}")
        print(f"平均每个客户端类别数: {np.mean([len(dist) for dist in client_class_dist]):.2f}")
        print(f"平均每个客户端空类数: {np.mean([len(vacant) for vacant in client_vacant_classes]):.2f}")

    else:
        raise NotImplementedError(f"Partition {partition} not implemented")

    return party2dataidx


# ----------------------------------------------------------------------
# DataLoader 构建函数（已添加 DermaMNIST 支持）
# ----------------------------------------------------------------------
def get_dataloader(args, dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom

        transform_train = []
        if args.auto_aug:
            transform_train.append(AutoAugment())

        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + '/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir + '/val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    # 新增 dermamnist 分支
    elif dataset == 'dermamnist':
        BANDWIDTH = 64
        train_transform = transforms.Compose([
            transforms.Resize((BANDWIDTH, BANDWIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((BANDWIDTH, BANDWIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_train_ds = DermaMNIST(split='train', transform=train_transform, download=True, root=datadir)
        if dataidxs is not None:
            train_ds = torch.utils.data.Subset(full_train_ds, dataidxs)
        else:
            train_ds = full_train_ds

        test_ds = DermaMNIST(split='test', transform=test_transform, download=True, root=datadir)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    else:
        raise NotImplementedError("dataset not implemented")

    return train_dl, test_dl, train_ds, test_ds