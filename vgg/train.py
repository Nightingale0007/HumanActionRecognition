import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vgg.vgg19_model import HumanVGG19


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def validate_model(model, data_loader, criterion, device):
    """在指定数据集上评估模型性能"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)

            predict = torch.argmax(output, dim=1)
            correct = (predict == label).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += data.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def test_model(model, test_loader, device):
    """在测试集上详细测试模型性能"""
    model.eval()
    test_correct = 0
    test_samples = 0
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            test_samples += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_accuracy = test_correct / test_samples

    # 计算每个类别的准确率
    class_accuracies = {}
    for i in range(len(class_correct)):
        if class_total[i] > 0:
            class_accuracies[test_loader.dataset.classes[i]] = class_correct[i] / class_total[i]

    return test_accuracy, class_accuracies, test_correct, test_samples


def train(train_data_path, val_data_path, test_data_path, epochs, learning_rate, model_save_path):
    """
    训练VGG19模型
    
    Args:
        train_data_path: 训练集路径
        val_data_path: 验证集路径  
        test_data_path: 测试集路径
        epochs: 训练迭代次数
        learning_rate: 学习率
        model_save_path: 模型保存路径
        
    Returns:
        model: 训练好的模型
        final_val_accuracy: 最终验证集准确率
        test_accuracy: 测试集准确率
        class_accuracies: 每个类别的测试准确率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing - training set with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation and test data preprocessing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    val_data = datasets.ImageFolder(root=val_data_path, transform=val_test_transform)
    test_data = datasets.ImageFolder(root=test_data_path, transform=val_test_transform)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    class_num = len(train_data.classes)
    label2index = train_data.class_to_idx
    index2label = {idx: label for label, idx in label2index.items()}

    print("Number of classes:", class_num)
    print("Class mapping:", label2index)
    print("Index to label mapping:", index2label)

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    model_path = None
    vgg19 = HumanVGG19(class_num, model_path).to(device)

    optimizer = optim.Adam(vgg19.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create directory for model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        vgg19.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = vgg19(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.argmax(output, dim=1)
            correct = (predict == label).sum().item()

            train_loss += loss.item()
            train_correct += correct
            train_samples += data.size(0)

            # Print every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                accuracy = correct / data.size(0)
                print(
                    f"Training - epoch: {epoch}, batch: {batch_idx + 1}/{len(train_loader)}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_samples

        avg_val_loss, avg_val_accuracy = validate_model(vgg19, val_loader, criterion, device)

        print(f"=== Epoch {epoch} Completed ===")
        print(f"Training - Average loss: {avg_train_loss:.4f}, Average accuracy: {avg_train_accuracy:.4f}")
        print(f"Validation - Average loss: {avg_val_loss:.4f}, Average accuracy: {avg_val_accuracy:.4f}")

        # Check early stopping
        early_stopping(avg_val_loss, vgg19)
        if early_stopping.early_stop:
            print("Early stopping: Training stopped")
            break

    print("Training completed!")

    # Final validation on validation set
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    final_val_loss, final_val_accuracy = validate_model(vgg19, val_loader, criterion, device)
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

    # Test on test set
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    test_accuracy, class_accuracies, test_correct, test_samples = test_model(vgg19, test_loader, device)
    print(f"Overall Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_samples})")
    
    print("\nPer-Class Test Accuracy:")
    for class_name, accuracy in class_accuracies.items():
        print(f"  {class_name}: {accuracy:.4f}")
    
    avg_class_accuracy = sum(class_accuracies.values()) / len(class_accuracies)
    print(f"\nAverage Class Accuracy: {avg_class_accuracy:.4f}")

    # Save the trained model
    torch.save(vgg19.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    return vgg19, final_val_accuracy, test_accuracy, class_accuracies


# 使用示例
if __name__ == "__main__":
    # 定义路径和参数
    train_path = "./data/UCF101_image/train"
    val_path = "./data/UCF101_image/val" 
    test_path = "./data/UCF101_image/test"
    
    # 调用训练函数
    trained_model, final_val_accuracy, test_accuracy, class_accuracies = train(
        train_data_path=train_path,
        val_data_path=val_path,
        test_data_path=test_path,
        epochs=50,
        learning_rate=0.001,
        model_save_path="./model/vgg19/trained_model.pth"
    )
    
    print(f"\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Average Class Accuracy: {sum(class_accuracies.values()) / len(class_accuracies):.4f}")