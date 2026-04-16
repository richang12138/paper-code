import time
from flcore.clients.clientdt import ClientDT
import torch.nn as nn
import torch
import numpy as np
import copy
import clip


class FedDT(object):
    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(args.model)
        self.teacher_model = copy.deepcopy(args.model)
        
        self.clip_model = None
        self.text_features = None
        self.preprocess = None
        self.label_names = None
        self.new_text_features = None
        
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
    
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times

        self.party2loaders_train = party2loaders
        self.party2loaders_test = test_dl
        
        self.global_class_counts = None
        self.global_class_ratio = None
        self.majority_classes = []
        self.normal_minority_classes = []
        self.rare_minority_classes = []
        self.class_avg_per_client = None
        
        self.total_train_samples = sum(len(loader.dataset) for loader in party2loaders.values())

        self.client_score_history = []

        self.set_clients(ClientDT, party2loaders)
        self.init_global_class_distribution()
        
        if self.num_clients > 0:
            self.class_avg_per_client = self.global_class_counts / self.num_clients
        else:
            self.class_avg_per_client = np.zeros(self.num_classes)
        print(f"每个类在每个客户端的平均样本数: {[round(avg, 2) for avg in self.class_avg_per_client]}")
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        
        self.random_state = np.random.RandomState(args.seed)
        
        if args.use_clip:
            self.load_clip_model()

    def init_global_class_distribution(self):
        self.global_class_counts = np.zeros(self.num_classes, dtype=int)
        for loader in self.party2loaders_train.values():
            for _, labels in loader:
                if isinstance(labels, torch.Tensor):
                    if labels.dim() > 1:
                        labels = labels.view(-1)
                    labels_np = labels.numpy()
                else:
                    if labels.ndim > 1:
                        labels = labels.flatten()
                    labels_np = labels
                for label in labels_np:
                    self.global_class_counts[label] += 1
        
        total_samples = self.global_class_counts.sum()
        self.global_class_ratio = self.global_class_counts / total_samples if total_samples > 0 else np.zeros(self.num_classes)
        
        print(f"全局总样本数: {total_samples}")
        print(f"全局类别样本数分布: {self.global_class_counts}")
        print(f"全局类别比例分布: {[round(r, 4) for r in self.global_class_ratio]}")

        sorted_classes = sorted(
            [(cls, count) for cls, count in enumerate(self.global_class_counts)],
            key=lambda x: x[1],
            reverse=True
        )
        sorted_cls = [cls for cls, _ in sorted_classes]
        sorted_counts = [count for _, count in sorted_classes]
        num_classes = len(sorted_cls)
        if num_classes == 0:
            return

        cumulative_ratio = []
        cumulative_sum = 0
        for count in sorted_counts:
            cumulative_sum += count
            cumulative_ratio.append(cumulative_sum / total_samples)
        print(f"累积样本占比（按类别排序）: {[round(r, 2) for r in cumulative_ratio]}")

        majority_threshold = 0.6
        normal_minority_threshold = 0.9
        majority_end = 0
        normal_end = 0

        for i in range(num_classes):
            if cumulative_ratio[i] >= majority_threshold:
                majority_end = i + 1
                break
        if majority_end == 0:
            majority_end = max(1, num_classes // 2)

        for i in range(majority_end, num_classes):
            if cumulative_ratio[i] >= normal_minority_threshold:
                normal_end = i + 1
                break
        if normal_end == 0:
            normal_end = majority_end + max(1, (num_classes - majority_end) // 2)

        majority_end = min(majority_end, num_classes)
        normal_end = min(normal_end, num_classes)
        self.majority_classes = sorted_cls[:majority_end]
        self.normal_minority_classes = sorted_cls[majority_end:normal_end]
        self.rare_minority_classes = sorted_cls[normal_end:]

        self.majority_classes = [cls for cls in self.majority_classes if self.global_class_counts[cls] > 0]
        self.normal_minority_classes = [cls for cls in self.normal_minority_classes if self.global_class_counts[cls] > 0]
        self.rare_minority_classes = [cls for cls in self.rare_minority_classes if self.global_class_counts[cls] > 0]

        majority_ratio = sum(self.global_class_ratio[cls] for cls in self.majority_classes)
        normal_ratio = sum(self.global_class_ratio[cls] for cls in self.normal_minority_classes)
        rare_ratio = sum(self.global_class_ratio[cls] for cls in self.rare_minority_classes)
        print(f"\n指数型长尾优化划分结果：")
        print(f"多数类（{len(self.majority_classes)}个）：{self.majority_classes}，样本占比：{majority_ratio:.2%}")
        print(f"普通少数类（{len(self.normal_minority_classes)}个）：{self.normal_minority_classes}，样本占比：{normal_ratio:.2%}")
        print(f"稀缺少数类（{len(self.rare_minority_classes)}个）：{self.rare_minority_classes}，样本占比：{rare_ratio:.2%}")

    def load_clip_model(self):
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.clip_model.eval()
        
        if self.dataset == 'cifar10':
            self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.dataset == 'cifar100':
            self.label_names = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
                'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
                'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
        # ================== 【核心修改】新增 DermaMNIST 支持 ==================
        elif self.dataset.lower() == 'dermamnist':
            # 使用医学专业术语（英文全称），确保 CLIP 能提取到准确的语义特征
            # 对应索引 0-6：akiec, bcc, bkl, df, mel, nv, vasc
            self.label_names = [
                'actinic keratosis',          # 0: 日光性角化病
                'basal cell carcinoma',       # 1: 基底细胞癌
                'benign keratosis',           # 2: 良性角化病
                'dermatofibroma',             # 3: 皮肤纤维瘤
                'melanoma',                   # 4: 黑色素瘤
                'melanocytic nevus',          # 5: 黑色素细胞痣
                'vascular lesion'             # 6: 血管病变
            ]
            print(f"已加载 DermaMNIST 专用 CLIP 文本标签: {self.label_names}")
        # =====================================================================
        else:
            self.label_names = [f"class_{i}" for i in range(self.num_classes)]
        
        text_inputs = clip.tokenize([f"a photo of a {c}" for c in self.label_names]).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_inputs)
        self.text_features = self.text_features.float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print(f"CLIP文本原型形状: {self.text_features.shape}")
        self.new_text_features = self.text_features

    def train(self):
        for round in range(self.global_rounds):
            s_t = time.time()
            
            if round == 0:
                print("\n===== 首轮训练前：向所有客户端发送类别平均值 =====")
                self.send_class_avg_to_all_clients()
            
            self.selected_clients = self.select_clients()
            self.send_models(round)

            print(f"\n-------------Round number: {round}-------------")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            print(f"-------------{current_time}-------------")

            for client in self.selected_clients:
                client.set_clip_info(self.clip_model, self.preprocess, self.text_features, self.label_names, self.new_text_features)
                client.train(self.party2loaders_train[client.id], round)
            
            self.collect_client_scores()
                
            self.receive_models(round)
            self.aggregate_parameters()
            
            self.teacher_model = copy.deepcopy(self.global_model)
            
            print("\nEvaluate aggregated global model")
            # 修改：此处返回宏观平均准确率和损失
            test_macro_acc, test_loss = self.compute_accuracy(self.global_model, self.party2loaders_test)
            print('>> Aggregated global model test accuracy (macro average): %f test loss: ' % test_macro_acc, test_loss)
            
            if round % 10 == 0 and hasattr(self.clients[0], 'no_exist_label'):
                print("\n当前轮次客户端信息：")
                for client in self.selected_clients:
                    score = client.score if hasattr(client, 'score') else 0.0
                    no_exist = client.no_exist_label.cpu().numpy() if hasattr(client, 'no_exist_label') else []
                    print(f"Client {client.id}: 空类={no_exist}, 综合评分={score:.4f}")
            
            self.rs_test_acc.append(test_macro_acc)
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest macro accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def send_class_avg_to_all_clients(self):
        assert (len(self.clients) > 0), "无客户端可发送信息"
        class_avg_tensor = torch.tensor(self.class_avg_per_client, dtype=torch.float32).to(self.device)
        for client in self.clients:
            start_time = time.time()
            client.set_global_info(
                global_class_dist=torch.tensor(self.global_class_ratio, dtype=torch.float32).to(self.device),
                rare_minority=self.rare_minority_classes,
                normal_minority=self.normal_minority_classes,
                majority=self.majority_classes,
                total_train_samples=self.total_train_samples,
                class_avg_per_client=class_avg_tensor
            )
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            print(f"已向客户端 {client.id} 发送类别平均值")

    def collect_client_scores(self):
        current_scores = []
        for client in self.clients:
            if hasattr(client, 'score') and client.score is not None:
                current_scores.append(client.score)
        if current_scores:
            self.client_score_history.append(current_scores)
            if len(self.client_score_history) > 50:
                self.client_score_history.pop(0)

    def compute_accuracy(self, model, dataloader):
        """
        计算宏观平均准确率（macro average accuracy）和平均损失。
        每类准确率之和除以类别数，未出现的类别准确率记为0。
        """
        was_training = False
        if model.training:
            model.eval()
            was_training = True

        class_correct = torch.zeros(self.num_classes, device=self.device)
        class_total = torch.zeros(self.num_classes, device=self.device)
        
        criterion = nn.CrossEntropyLoss()
        loss_collector = []
        
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(self.device), target.to(dtype=torch.int64).to(self.device)
                # 兼容 DermaMNIST 二维标签
                if target.dim() > 1:
                    target = target.view(-1)
                
                out = model(x)
                loss = criterion(out, target)
                loss_collector.append(loss.item())
                
                _, pred_label = torch.max(out.data, 1)
                
                # 统计每个类别的正确数和总数
                for label, pred in zip(target, pred_label):
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        avg_loss = sum(loss_collector) / len(loss_collector) if loss_collector else 0.0
        
        # 计算每类准确率（未出现的类别准确率为0）
        class_acc = []
        for i in range(self.num_classes):
            if class_total[i] > 0:
                class_acc.append(class_correct[i].item() / class_total[i].item())
            else:
                class_acc.append(0.0)
        
        # 打印每类准确率（便于监控）
        print(f"每类准确率: {[round(acc, 4) for acc in class_acc]}")
        
        # 宏观平均准确率 = 每类准确率之和 / 类别数
        macro_acc = sum(class_acc) / self.num_classes if self.num_classes > 0 else 0.0

        if was_training:
            model.train()

        return macro_acc, avg_loss

    def set_clients(self, clientObj, party2loaders):
        for i in range(self.num_clients):
            dataload = party2loaders[i]
            client = clientObj(self.args, id=i, train_samples=len(dataload.dataset))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = self.random_state.choice(
                range(self.num_join_clients, self.num_clients + 1), 
                1, 
                replace=False
            )[0]
        total_selected = self.current_num_join_clients
        print(f"\n本轮选择客户端数量: {total_selected}")
        selected_clients = list(self.random_state.choice(self.clients, total_selected, replace=False))
        selected_ids = sorted([c.id for c in selected_clients])
        print(f"选中的客户端ID: {selected_ids}")
        return selected_clients

    def send_models(self, current_round):
        assert (len(self.clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            global_info = {
                'global_class_dist': torch.tensor(self.global_class_ratio, dtype=torch.float32).to(self.device),
                'rare_minority': self.rare_minority_classes,
                'normal_minority': self.normal_minority_classes,
                'majority': self.majority_classes,
                'total_train_samples': self.total_train_samples
            }
            client.set_global_info(**global_info)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self, current_round):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_weight = 0.0
        all_selected_clients = self.selected_clients
        print(f"✅ 本轮所有 {len(all_selected_clients)} 个选中客户端均参与聚合（无过滤）")
        for client in all_selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
            client_weight = client.train_samples * client.score
            self.uploaded_weights.append(client_weight)
            total_weight += client_weight
        self.uploaded_weights = [w / total_weight for w in self.uploaded_weights]
        print(f"\n客户端权重分布（ID: 权重: 评分）:")
        for idx, (client_id, weight, client) in enumerate(zip(self.uploaded_ids, self.uploaded_weights, all_selected_clients)):
            print(f"Client {client_id}: 权重={weight:.4f}, 评分={client.score:.4f}")

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        global_model_w = self.global_model.state_dict()
        temp = True
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_model_w = client_model.state_dict()
            if temp:
                for key in client_model_w:
                    global_model_w[key] = client_model_w[key] * w
                temp = False
            else:
                for key in client_model_w:
                    global_model_w[key] += client_model_w[key] * w
        self.global_model.load_state_dict(global_model_w)

    def check_done(self, acc_lss, top_cnt=100):
        if len(acc_lss[0]) < top_cnt:
            return False
        for acc in acc_lss:
            if max(acc[-top_cnt:]) - min(acc[-top_cnt:]) > 1e-4:
                return False
        return True
