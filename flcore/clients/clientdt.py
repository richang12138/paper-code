import torch
import torch.nn as nn
import numpy as np
import math
import time
import os
import copy
import torch.nn.functional as F
from torchvision import transforms

class SupConLoss_text(nn.Module):
    """增强版：融合样本间对比和CLIP原型对比（包含空类）"""
    def __init__(self, device, temperature=0.07, num_classes=10):
        super(SupConLoss_text, self).__init__()
        self.device = device
        self.temperature = temperature
        self.num_classes = num_classes
        
    def forward(self, features, labels, text_features):
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        
        # 1. 样本间对比损失（仅针对非空类样本）
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        mask = torch.zeros_like(sim_matrix)
        for i in range(batch_size):
            mask[i, labels == labels[i]] = 1.0  # 同类样本为正例
        mask.fill_diagonal_(0)  # 排除自身
        pos_pairs = torch.sum(sim_matrix * mask, dim=1)
        neg_pairs = torch.sum(sim_matrix * (1 - mask), dim=1)
        sample_loss = -torch.mean(torch.log(pos_pairs / (neg_pairs + 1e-8) + 1e-8))
        
        # 2. 样本与CLIP文本原型的对比损失（包含空类原型）
        text_features = F.normalize(text_features, p=2, dim=1)
        proto_sim = torch.matmul(features, text_features.T) / self.temperature  # [batch_size, num_classes]
        pos_proto = proto_sim[torch.arange(batch_size), labels]  # 同类原型（非空类）
        neg_proto = torch.logsumexp(proto_sim, dim=1)  # 所有异类原型（包含空类和其他非空类）
        proto_loss = -torch.mean(pos_proto - neg_proto)
        
        # 融合两种损失
        total_loss = sample_loss + 0.5 * proto_loss
        return total_loss

class KDLoss(nn.Module):
    '''知识蒸馏损失'''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return kd_loss

class ClientDT(object):
    def __init__(self, args, id, train_samples, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device 
        self.id = id
        self.num_classes = args.num_classes
        self.train_samples = train_samples  # 客户端数据量
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.weight_decay = args.weight_decay
        self.weight_decay = 0.0
        
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        
        self.num_per_class = None
        self.prior = None
        self.no_exist_label = None  # 空类标签（保留但不再单独用于蒸馏）
        self.exist_label = None     # 非空类标签（保留但不再用于筛选薄弱类）
        self.teacher_model = None
        self.beta_y = None
        self.beta = args.beta
        self.lamda = args.lamda
        self.global_epoch = 0
        
        # CLIP相关属性
        self.clip_model = None
        self.preprocess = None
        self.text_features = None  # 包含所有类的CLIP文本原型（含空类）
        self.label_names = None
        self.new_text_features = None  # 保留用于兼容原有代码
        
        # 评分机制相关属性（新增/保留）
        self.global_class_dist = None  # 全局类别分布（由服务器下发）
        self.rare_minority_classes = None  # 稀缺少数类（由服务器定义）
        self.normal_minority_classes = None  # 普通少数类（由服务器定义）
        self.majority_classes = None  # 多数类（由服务器定义）
        self.client_dist = None  # 客户端类别分布
        self.score = 0.0  # 综合评分（用于调整聚合权重）
        self.total_train_samples = None  # 全局总样本数（由服务器下发）
        
        # ---------------------- 新增：存储服务器下发的每个类平均数 ----------------------
        self.class_avg_per_client = None  # 每个类在客户端的平均数量（服务器首轮下发）
        # -----------------------------------------------------------------------------
        
        # ---------------------- 新评分系统配置参数 ----------------------
        self.valid_cover_thresh = 5  # 类别“有效覆盖”的最小样本数（避免1个样本算覆盖）
        self.minority_perf_weight = 0.6  # 少数类准确率在“本地性能”中的占比
        self.majority_perf_weight = 0.4  # 多数类准确率在“本地性能”中的占比
        self.dim_weights = {  # 各维度权重（总和为1.0）
            "class_coverage": 0.3,    # 类别覆盖与代表性
            "local_performance": 0.3, # 本地模型性能
            "dist_complement": 0.2,   # 分布互补性
            "data_fairness": 0.2      # 数据量公平性
        }
        # ----------------------------------------------------------------
        
        self.loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(T=args.T)
        self.contras_criterion = SupConLoss_text(args.device, args.ins_temp, args.num_classes)

    def set_clip_info(self, clip_model, preprocess, text_features, label_names, new_text_features=None):
        """设置CLIP信息，text_features包含所有类的原型（含空类）"""
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.text_features = text_features  # [num_classes, feature_dim] 包含所有类
        self.label_names = label_names
        self.new_text_features = new_text_features

    # ---------------------- 修改：接收服务器下发的每个类平均数 ----------------------
    def set_global_info(self, global_class_dist, rare_minority, normal_minority, majority, total_train_samples, class_avg_per_client=None):
        """由服务器设置全局分布和少数类定义，新增class_avg_per_client参数接收类平均数"""
        self.global_class_dist = global_class_dist.to(self.device)
        self.rare_minority_classes = rare_minority
        self.normal_minority_classes = normal_minority
        self.majority_classes = majority
        self.total_train_samples = total_train_samples  # 接收全局总样本数
        # 存储首轮下发的每个类平均数（转换为设备兼容格式）
        if class_avg_per_client is not None:
            self.class_avg_per_client = class_avg_per_client.to(self.device)
            print(f"Client {self.id} 接收每个类平均数量: {[round(x.item(), 2) for x in self.class_avg_per_client]}")
        # -----------------------------------------------------------------------------

    def compute_client_distribution(self, trainloader):
        """计算客户端类别分布（比例），兼容 DermaMNIST 二维标签"""
        dist = torch.zeros(self.num_classes, device=self.device)
        total = 0.0
        for _, labels in trainloader:
            # 处理 DermaMNIST 标签形状 (batch_size, 1)
            if labels.dim() > 1:
                labels = labels.view(-1)
            for label in labels:
                dist[label.item()] += 1
                total += 1
        if total > 0:
            dist /= total
        self.client_dist = dist
        return dist

    # ---------------------- 新评分系统：4个维度辅助方法 ----------------------
    def compute_local_performance(self, trainloader):
        """辅助1：计算本地模型性能（区分少数类/多数类准确率），兼容二维标签"""
        self.model.eval()
        stat = {
            "rare_minority": {"correct": 0, "total": 0},
            "normal_minority": {"correct": 0, "total": 0},
            "majority": {"correct": 0, "total": 0}
        }

        with torch.no_grad():
            for x, y in trainloader:
                # 适配输入格式（单输入/多输入）
                if type(x) == type([]):
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)
                # 展平标签
                if y.dim() > 1:
                    y = y.view(-1)
                y = y.long()
                
                outputs = self.model(x_in)
                _, preds = torch.max(outputs, 1)

                # 按类别类型统计
                for label, pred in zip(y, preds):
                    label = label.item()
                    if label in self.rare_minority_classes:
                        stat["rare_minority"]["total"] += 1
                        if pred == label:
                            stat["rare_minority"]["correct"] += 1
                    elif label in self.normal_minority_classes:
                        stat["normal_minority"]["total"] += 1
                        if pred == label:
                            stat["normal_minority"]["correct"] += 1
                    elif label in self.majority_classes:
                        stat["majority"]["total"] += 1
                        if pred == label:
                            stat["majority"]["correct"] += 1

        # 计算各类别准确率（避免除零）
        rare_acc = stat["rare_minority"]["correct"] / max(1, stat["rare_minority"]["total"])
        normal_acc = stat["normal_minority"]["correct"] / max(1, stat["normal_minority"]["total"])
        major_acc = stat["majority"]["correct"] / max(1, stat["majority"]["total"])

        # 综合少数类准确率（稀缺类权重更高）
        minority_acc = 0.6 * rare_acc + 0.4 * normal_acc
        # 本地性能总分（少数类占比更高）
        total_perf = self.minority_perf_weight * minority_acc + self.majority_perf_weight * major_acc
        
        self.model.train()
        return round(total_perf, 4)

    def compute_class_coverage(self):
        """辅助2：计算类别覆盖与代表性（强化少数类权重）"""
        if self.client_dist is None:
            return 0.0

        # 计算“有效覆盖”（样本数≥阈值才算有效）
        def _get_valid_cover(class_list):
            valid_count = 0
            for cls in class_list:
                # 客户端该类实际样本数 = 客户端总样本数 × 分布比例
                cls_samples = self.train_samples * self.client_dist[cls]
                if cls_samples >= self.valid_cover_thresh:
                    valid_count += 1
            total_count = max(1, len(class_list))  # 避免除零
            return valid_count / total_count

        # 各类别类型的覆盖得分（稀缺类权重最高）
        rare_cover = _get_valid_cover(self.rare_minority_classes)
        normal_cover = _get_valid_cover(self.normal_minority_classes)
        major_cover = _get_valid_cover(self.majority_classes)

        # 综合覆盖得分
        total_cover = (0.5 * rare_cover + 0.3 * normal_cover + 0.2 * major_cover)
        return round(total_cover, 4)

    def compute_dist_complement(self):
        """辅助3：计算分布互补性（KL散度归一化）"""
        if self.client_dist is None or self.global_class_dist is None:
            return 0.0

        eps = 1e-8
        # KL散度：客户端分布 || 全局分布（衡量差异）
        kl_div = torch.sum(self.client_dist * torch.log(self.client_dist / (self.global_class_dist + eps) + eps)).item()
        # 归一化（理论上限为log(类别数)）
        max_kl = math.log(self.num_classes)
        normalized_kl = min(1.0, kl_div / max_kl)
        # 互补性得分：差异越大，得分越高（限制最大为0.9避免极端值）
        complement_score = min(0.9, 1.0 - normalized_kl)
        return round(complement_score, 4)

    def compute_data_fairness(self):
        """辅助4：计算数据量公平性（平方根缓和差异）"""
        if self.total_train_samples is None or self.args.num_clients == 0:
            return 0.1

        # 全局平均样本数
        avg_samples = self.total_train_samples / self.args.num_clients
        # 平方根修正（小客户端得分更高）
        if avg_samples == 0:
            fairness_score = 0.1
        else:
            sample_ratio = self.train_samples / avg_samples
            fairness_score = min(1.0, math.sqrt(sample_ratio))
        # 保证最小得分
        return round(max(0.1, fairness_score), 4)

    # ---------------------- 新评分系统核心方法 ----------------------
    def compute_metrics(self, trainloader):
        """新综合评分：整合4个维度，输出最终评分"""
        # 1. 计算各维度得分
        class_coverage = self.compute_class_coverage()
        local_performance = self.compute_local_performance(trainloader)
        dist_complement = self.compute_dist_complement()
        data_fairness = self.compute_data_fairness()

        # 2. 按权重加权求和
        total_score = (
            self.dim_weights["class_coverage"] * class_coverage +
            self.dim_weights["local_performance"] * local_performance +
            self.dim_weights["dist_complement"] * dist_complement +
            self.dim_weights["data_fairness"] * data_fairness
        )

        # 3. 异常值处理（限制在[0.1, 1.0]）
        total_score = max(0.1, min(1.0, total_score))
        self.score = round(total_score, 4)

        # 4. 打印评分拆解（调试用）
        print(f"\nClient {self.id} 评分拆解：")
        print(f"  - 类别覆盖得分：{class_coverage:.4f} (权重{self.dim_weights['class_coverage']})")
        print(f"  - 本地性能得分：{local_performance:.4f} (权重{self.dim_weights['local_performance']})")
        print(f"  - 分布互补得分：{dist_complement:.4f} (权重{self.dim_weights['dist_complement']})")
        print(f"  - 数据公平得分：{data_fairness:.4f} (权重{self.dim_weights['data_fairness']})")

        return self.score

    def train(self, data_this_client, round):
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                        lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        start_time = time.time()
        trainloader = data_this_client
        self.global_epoch = round
        
        # 初始化类别分布信息（兼容 DermaMNIST 二维标签）
        if self.num_per_class is None:
            self.num_per_class = [0. for i in range(self.num_classes)]
            for _, labels in trainloader:
                # 处理二维标签
                if labels.dim() > 1:
                    labels = labels.view(-1)
                for label in labels:
                    self.num_per_class[label.item()] += 1         
            self.num_per_class = torch.Tensor(self.num_per_class).float().to(self.device)
            self.prior = self.num_per_class / self.num_per_class.sum()
            self.prior = self.prior.to(self.device)
            self.beta_y = (self.num_per_class / self.num_per_class.max()).pow(0.05)
            self.no_exist_label = torch.where(self.prior == 0)[0].to(self.device)  # 空类标签（保留）
            self.exist_label = torch.where(self.prior != 0)[0].to(self.device)     # 非空类标签（保留）
            self.no_exist_label = torch.Tensor(self.no_exist_label).int().to(self.device)
            self.exist_label = torch.Tensor(self.exist_label).int().to(self.device)
            self.exist_prior = self.prior[self.exist_label]
        
        # 计算客户端分布（原有逻辑不变）
        self.compute_client_distribution(trainloader)
        
        # 【修改】调用新的compute_metrics，传入trainloader计算本地性能
        if self.global_class_dist is not None:
            self.compute_metrics(trainloader)
            print(f"Client {self.id} 综合评分: {self.score:.4f} (数据量: {self.train_samples})")
        
        # ---------------------- 核心修改1：筛选“本地数量少于平均数”的类别 ----------------------
        # 确保已接收类平均数（服务器首轮下发，后续轮次复用）
        under_avg_labels = torch.tensor([], dtype=torch.int64).to(self.device)
        if self.class_avg_per_client is not None:
            # 遍历所有类，筛选样本数 < 该类平均数的类别
            under_avg_mask = self.num_per_class < self.class_avg_per_client
            under_avg_labels = torch.where(under_avg_mask)[0].to(self.device)
            # 转换为int64类型以匹配蒸馏损失输入要求
            under_avg_labels = under_avg_labels.int().to(self.device)
        print(f"Client {self.id} 本地数量少于平均数的类别: {under_avg_labels.cpu().numpy()} (共{len(under_avg_labels)}个)")
        # -----------------------------------------------------------------------------
        
        self.model.train()
        print(f"\n-------------client: {self.id}-------------")    
        max_local_epochs = self.local_epochs
    
        for step in range(max_local_epochs):
            epoch_loss_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 展平标签
                if y.dim() > 1:
                    y = y.view(-1)
                y = y.long()   
                             
                self.optimizer.zero_grad()                
                output = self.model(x)
                
                # 计算原始损失（原有逻辑不变）
                LADE_loss = self.new_LADEloss(output, y)
                Prior_CELoss = self.PriorCELoss(output, y)
                
                # CLIP知识蒸馏（包含空类和非空类，原有逻辑不变）
                clip_loss, feature_output = self.clip_knowledge_distillation(x, output, y)
                
                # ---------------------- 核心修改2：蒸馏目标替换为“少于平均数的类别” ----------------------
                # 原逻辑：no_exist_loss + weak_exist_loss；新逻辑：仅对under_avg_labels蒸馏
                teach_output = self.teacher_model(x).detach()
                under_avg_loss = self.distillation_loss(output, teach_output, under_avg_labels) if len(under_avg_labels) > 0 else 0.0
                VLS_loss = under_avg_loss  # 直接使用少于平均数类别的蒸馏损失
                # -----------------------------------------------------------------------------
                
                # 对比学习损失（包含空类原型，原有逻辑不变）
                contrast_loss = 0.0
                if self.text_features is not None and feature_output is not None:
                    contrast_loss = self.contras_criterion(feature_output, y, self.text_features)
                
                # 组合损失（原有逻辑不变，仅VLS_loss来源修改）
                loss = Prior_CELoss + LADE_loss + VLS_loss * self.lamda + clip_loss * self.args.clip_alpha
                if contrast_loss > 0:
                    loss += self.args.contrast_alpha * contrast_loss
                
                loss.backward()
                self.optimizer.step()
                epoch_loss_collector.append(loss.item())
                
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            print('Epoch: %d Loss: %f' % (step, epoch_loss))
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def set_parameters(self, model):
        """原有逻辑不变：设置全局模型和教师模型"""
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
        self.teacher_model = copy.deepcopy(model)      
    
    def new_LADEloss(self, y_pred, target): 
        """原有逻辑不变：LADE损失计算"""
        cls_weight = (self.num_per_class.float() / torch.sum(self.num_per_class.float())).to(self.device)
        balanced_prior = torch.tensor(1. / self.num_classes, device=self.device).float()
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T * (target != torch.arange(0, self.num_classes).view(-1, 1).type_as(target))
        N = pred_spread.size(-1)
        second_term = torch.logsumexp(pred_spread, -1) - np.log(N)
        loss = - torch.sum( (- second_term ) * cls_weight)
        return loss * 0.005
    
    def PriorCELoss(self, output, y):
        """原有逻辑不变：带先验的交叉熵损失"""
        logits = output + torch.log(self.prior + 1e-9)
        loss = self.loss(logits, y)
        return loss

    def distillation_loss(self, output, teach_output, target_labels):
        """通用蒸馏损失，兼容空标签列表"""
        if len(target_labels) == 0:
            return torch.tensor(0.0, device=self.device)
        output_log_soft = torch.nn.functional.log_softmax(output[:, target_labels], dim=1)
        teach_output_soft = torch.nn.functional.softmax(teach_output[:, target_labels], dim=1)
        kl = nn.KLDivLoss(reduction='batchmean')
        return kl(output_log_soft, teach_output_soft)
    
    # 保持原有VLSloss方法兼容（未使用但保留）
    def VLSloss(self, output, teach_output, no_exist_label, exist_label, exist_prior, y):
        return self.distillation_loss(output, teach_output, no_exist_label)
    
    def clip_knowledge_distillation(self, x, model_output, target):
        """原有逻辑不变：CLIP知识蒸馏"""
        if self.clip_model is None:
            return torch.tensor(0.0, device=self.device), None
            
        x_clip = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x_clip)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # CLIP计算所有类的logits（包含空类）
        clip_logits = 100.0 * image_features @ self.text_features.T  # [batch_size, num_classes]
        
        # 蒸馏损失包含所有类（自然涵盖空类和非空类）
        kd_loss = self.kd_loss(model_output, clip_logits)
        
        # 提取特征用于对比学习
        feature_output = None
        if hasattr(self.model, 'feature_extractor'):
            feature_output = self.model.feature_extractor(x)
        elif isinstance(self.model, nn.Sequential) and len(self.model) > 1:
            feature_output = self.model[:-1](x)
        
        return kd_loss, feature_output
