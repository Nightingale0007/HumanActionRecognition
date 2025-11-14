import torch
import random
import numpy as np
from torch import nn, optim
import c3d.C3D_model as C3D_model
import os
from datetime import datetime
import socket
import timeit
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from c3d.dataset import VideoDataset

def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(num_epochs, num_classes, lr, device, model_save_path, train_dataloader, val_dataloader, test_dataloader):
    # 设置随机种子
    set_seed()
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # C3D模型实例化 - 每次都重新创建
    model = C3D_model.C3D(num_classes, pretrained=True)
    
    # 定义模型的损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器 - 每次都重新创建
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 定义学习率的更新策略 - 每次都重新创建
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 将模型和损失函数放入到设备中
    model.to(device)
    criterion.to(device)

    # 开始模型的训练
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    # 训练早停设置 - 每次都重新初始化
    best_model_wts = None  # 改为保存模型权重而不是模型对象
    best_test_loss = float('inf')
    early_stop_count = 0
    
    # 记录开始训练的信息
    print(f"Commencing training, total number of epochs: {num_epochs}")
    print(f"Training set size: {trainval_sizes['train']}, Verification set size: {trainval_sizes['val']}")

    # 开始训练
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            for inputs, labels in tqdm(trainval_loaders[phase], desc=f'{phase} Epoch {epoch+1}'):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == "train":
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                labels = labels.long()
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            stop_time = timeit.default_timer()

            print(f"[{phase.upper()}] Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"Execution time: {stop_time - start_time:.2f}秒")
            print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}\n")

            if phase == "val":
                if epoch_loss < best_test_loss:
                    # 保存模型权重而不是模型对象
                    best_model_wts = model.state_dict().copy()
                    best_test_loss = epoch_loss
                    early_stop_count = 0
                    print(f"*** New best model! Validation loss: {best_test_loss:.4f} ***")
                else:
                    early_stop_count += 1
                    print(f"Early stop count: {early_stop_count}/3")

        if early_stop_count >= 3:
            print(f"Parked early Epoch {epoch + 1}")
            break

    # 加载最佳模型权重进行测试
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # 开始模型的测试
    model.eval()
    running_corrects = 0.0
    
    print("Commencing testing...")
    for inputs, labels in tqdm(test_dataloader, desc='Testing'):
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        
        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_acc = running_corrects.double() / test_size
    print(f"Test accuracy: {epoch_acc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), model_save_path)  # 只保存权重，不是整个模型
    print(f"The model has been saved to: {model_save_path}")
    
    return model