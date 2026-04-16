#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverdt import FedDT
from flcore.servers.servermr import FedMR
from flcore.servers.serverntd import FedNTD
from flcore.servers.serversam import FedSAM
from flcore.servers.serverlogitcal import FedLogitCal
from flcore.servers.serverrs import FedRS
from flcore.servers.serverexp import FedEXP
from flcore.servers.serverprox import FedProx
from flcore.servers.servermoon import MOON

from flcore.clients.clientdt import clientdt

from flcore.trainmodel.models import *
from flcore.trainmodel.resnetcifar import *
from flcore.trainmodel.mobilenetv2 import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from data.pacs_dataset import *
from data.meta_dataset import *
from data.generate_mnist import *
from dataset_utils import partition_data, get_dataloader

import clip
from torchvision import transforms

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(10)

vocab_size = 98635
max_len = 200
emb_dim = 32

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_name = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "dnn":
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        elif model_str == "resnet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "resnet32":
            args.model = resnet32(num_classes=args.num_classes).to(args.device)
        elif model_str == "mobilenetv2":
            args.model = mobilenetv2(num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        if args.dataset == 'mnist':
            party2loaders, global_train_dl, test_dl = generate_mnist(
                args.datadir, args.num_classes, args.num_clients,
                niid=True, balance=False, partition=args.partition, alpha=args.alpha
            )
        else:
            party2dataidx = partition_data(
                args.dataset, args.datadir, args.partition, args.num_clients, alpha=args.alpha
            )

            party2loaders = {}
            party2loaders_ds = {}
            datadistribution = np.zeros((args.num_clients, args.num_classes, 2))

            for party_id in range(args.num_clients):
                train_dl_local, _, train_ds_local, _ = get_dataloader(
                    args, args.dataset, args.datadir,
                    args.batch_size, args.batch_size, party2dataidx[party_id]
                )
                party2loaders[party_id] = train_dl_local
                party2loaders_ds[party_id] = train_ds_local
                for k in range(args.num_classes):
                    datadistribution[party_id][k][0] = k

                # ---------- 修改开始：处理 DermaMNIST 二维标签 ----------
                all_labels = np.empty((0,), dtype=np.int64)
                for data, targets in party2loaders[party_id]:
                    # 如果标签是二维 (batch_size, 1)，展平为一维
                    if targets.dim() > 1:
                        labels = targets.view(-1).numpy()
                    else:
                        labels = targets.numpy()
                    all_labels = np.concatenate((all_labels, labels), axis=0)
                # ---------- 修改结束 ----------

                uniq_val, uniq_count = np.unique(all_labels, return_counts=True)
                for j, c in enumerate(uniq_val.tolist()):
                    datadistribution[party_id][c][1] = uniq_count[j]

            np.set_printoptions(threshold=np.inf)
            print("客户端数据分布（行：客户端ID，列：类别ID，值：样本数）:")
            print(datadistribution[..., 1])

            global_train_dl, test_dl, _, _ = get_dataloader(
                args, args.dataset, args.datadir,
                train_bs=args.batch_size, test_bs=args.batch_size
            )

        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedDT":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            if hasattr(args.model, 'feature_extractor'):
                if "mnist" in args.dataset:
                    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
                elif "cifar" in args.dataset:
                    dummy_input = torch.randn(1, 3, 32, 32).to(args.device)
                else:
                    dummy_input = torch.randn(1, 3, 224, 224).to(args.device)
                with torch.no_grad():
                    feature_dim = args.model.feature_extractor(dummy_input).shape[1]
                args.feature_dim = feature_dim
            else:
                args.feature_dim = 512
            server = FedDT(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedMR":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedMR(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedNTD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedNTD(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedLogitCal":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLogitCal(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedSAM":
            server = FedSAM(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedRS":
            server = FedRS(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedEXP":
            server = FedEXP(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i, party2loaders, global_train_dl, test_dl)

        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ed', "--weight_decay", type=float, default=1e-5)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False)
    parser.add_argument('-nc', "--num_clients", type=int, default=2)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-pv', "--prev", type=int, default=0)
    parser.add_argument('-dd', '--datadir', type=str, required=False, default="./data/")

    parser.add_argument('-tth', "--time_threthold", type=float, default=10000)
    parser.add_argument('-bt', "--beta", type=float, default=0.005)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    parser.add_argument('-mu', "--mu", type=float, default=0.001)

    parser.add_argument('-pro_d', "--proj_dim", type=int, default=256)
    parser.add_argument('-tem', "--temperature", type=float, default=0.5)
    parser.add_argument('-use_prod', "--use_proj_head", type=bool, default=True)

    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-partition', '--partition', type=str, default='noniid')
    parser.add_argument('-aug', '--auto_aug', type=bool, default=True)

    parser.add_argument('-tau', "--tau", type=float, default=0.001)
    parser.add_argument('-mom', "--momentum", type=float, default=0.9)
    parser.add_argument('-rho', "--rho", type=float, default=1.0)
    parser.add_argument('-cal_tem', "--calibration_temp", type=float, default=0.1)
    parser.add_argument('-rs', "--restricted_strength", type=float, default=0.5)
    parser.add_argument('-eps', "--eps", type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('-wda', "--weak_distill_alpha", type=float, default=0.1)

    parser.add_argument('-uc', "--use_clip", type=bool, default=True)
    parser.add_argument('-ca', "--clip_alpha", type=float, default=1.0)
    parser.add_argument('--ins_temp', type=float, default=0.07)
    parser.add_argument('--T', type=float, default=4.0)
    parser.add_argument('--contrast_alpha', type=float, default=1.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda不可用，自动切换到cpu\n")
        args.device = "cpu"

    print("=" * 50)
    print("算法: {}".format(args.algorithm))
    print("本地批次大小: {}".format(args.batch_size))
    print("本地训练轮数: {}".format(args.local_epochs))
    print("本地学习率: {}".format(args.local_learning_rate))
    print("权重衰减: {}".format(args.weight_decay))
    print("客户端总数: {}".format(args.num_clients))
    print("每轮参与客户端比例: {}".format(args.join_ratio))
    print("运行次数: {}".format(args.times))
    print("数据集: {}".format(args.dataset))
    print("类别数: {}".format(args.num_classes))
    print("模型 backbone: {}".format(args.model))
    print("使用设备: {}".format(args.device))
    print("自动停止: {}".format(args.auto_break))
    if not args.auto_break:
        print("全局轮数: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("CUDA设备ID: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("权重衰减系数: {}".format(args.weight_decay))
    print("非独立同分布程度: {}".format(args.alpha))
    print("是否使用自动增强: {}".format(args.auto_aug))

    if args.algorithm == "FedDT":
        print(f"薄弱存在类蒸馏系数: {args.weak_distill_alpha}")

    if args.use_clip:
        print("使用CLIP进行语义监督（包含空类原型）")
        print(f"CLIP损失权重: {args.clip_alpha}")
        print(f"对比损失权重（含空类）: {args.contrast_alpha}")
        print(f"实例对比温度: {args.ins_temp}")
    else:
        print("不使用CLIP")

    if args.algorithm == "FedProx":
        print("近端项系数: {}".format(args.mu))
    elif args.algorithm == "MOON":
        print("MOON损失系数: {}".format(args.mu))
        print("投影头维度: {}".format(args.proj_dim))
        print("对比损失温度: {}".format(args.temperature))
        print("是否使用投影头: {}".format(args.use_proj_head))
    elif args.algorithm == "FedSAM":
        print("动量系数: {}".format(args.momentum))
        print("rho参数: {}".format(args.rho))
    elif args.algorithm == "FedLogitCal":
        print("校准温度: {}".format(args.calibration_temp))
    elif args.algorithm == "FedRS":
        print("限制强度: {}".format(args.restricted_strength))
    elif args.algorithm == "FedEXP":
        print("epsilon: {}".format(args.eps))
    elif args.algorithm == "FedNTD":
        print("NTD损失系数: {}".format(args.beta))

    print("=" * 50)

    run(args)

    current_struct_time1 = time.localtime(time.time())
    formatted_time1 = time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time1)
    print(f"程序结束时间: {formatted_time1}")
    print(f"总运行时间: {time.time() - total_start:.2f}秒")