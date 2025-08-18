"""
由deepseek生成的通用pytorch训练模板，放这这里用于对比学习。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import os
import time
from tqdm import tqdm

# 设置随机种子保证可复现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# ==================== 配置参数 ====================
class Config:
    # 数据集参数
    dataset_path = './data'
    batch_size = 128
    num_workers = 4  # 数据加载的线程数
    
    # 训练参数
    epochs = 50
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    
    # 模型保存
    save_dir = './checkpoints'
    best_model_name = 'best_model.pth'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# ==================== 数据加载 ====================
def get_dataloaders():
    """创建训练集和测试集的数据加载器"""
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

    train_set = CIFAR10(root=Config.dataset_path, train=True,
                        download=True, transform=transform_train)
    test_set = CIFAR10(root=Config.dataset_path, train=False,
                       download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=Config.batch_size,
                              shuffle=True, num_workers=Config.num_workers)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size,
                             shuffle=False, num_workers=Config.num_workers)
    
    return train_loader, test_loader

# ==================== 模型定义 ====================
class SampleModel(nn.Module):
    """示例模型结构，实际使用时替换为你的模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== 训练函数 ====================
def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader):
    """模型训练主循环"""
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # 创建模型保存目录
    os.makedirs(Config.save_dir, exist_ok=True)
    
    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # 计算训练损失
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        # 验证阶段
        val_loss, val_acc = validate_model(model, criterion, test_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'loss': val_loss,
            }, os.path.join(Config.save_dir, Config.best_model_name))
        
        # 打印统计信息
        time_elapsed = time.time() - start_time
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Time elapsed: {time_elapsed:.0f}s")
    
    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")
    return history

def validate_model(model, criterion, test_loader):
    """模型验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validating"):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(test_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

# ==================== 主函数 ====================
def main():
    # 初始化数据加载器
    train_loader, test_loader = get_dataloaders()
    
    # 初始化模型
    model = SampleModel().to(Config.device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr, 
                          momentum=Config.momentum, weight_decay=Config.weight_decay)
    
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 训练模型
    history = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader)
    
    # 可选：绘制训练曲线
    # plot_training_history(history)

if __name__ == "__main__":
    main()