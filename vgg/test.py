import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vgg.vgg19_model import HumanVGG19


def test_model(model, test_loader, device):
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

    class_accuracies = {}
    for i in range(len(class_correct)):
        if class_total[i] > 0:
            class_accuracies[test_loader.dataset.classes[i]] = class_correct[i] / class_total[i]

    return test_accuracy, class_accuracies, test_correct, test_samples


def tested_model(tested_model, test_date):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = datasets.ImageFolder(root="./data/UCF101_image/test", transform=test_transform)

    print(f"Test set size: {len(test_data)}")

    class_num = len(test_data.classes)
    print("Number of classes:", class_num)
    print("Classes:", test_data.classes)

    test_loader = DataLoader(test_data, batch_size=1,shuffle=False)
    print(f"Test batches: {len(test_loader)}")

    model_path = "./model/vgg19/checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        model_path = "./model/vgg19/final_model.pth"
        if not os.path.exists(model_path):
            print(f"Model not found. Please train the model first.")
            return

    vgg19 = HumanVGG19(class_num, model_path=None).to(device)

    if model_path.endswith("best_model.pth"):
        checkpoint = torch.load(model_path)
        vgg19.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded (from epoch {checkpoint['epoch']})")
    else:
        vgg19.load_state_dict(torch.load(model_path))
        print("Final model loaded successfully!")

    test_accuracy, class_accuracies, test_correct, test_samples = test_model(vgg19, test_loader, device)

    print("=== Test Results ===")
    print(f"Overall Test Accuracy: {test_accuracy:.4f} ({test_correct}/{test_samples})")

    print("\n=== Per-Class Accuracy ===")
    for class_name, accuracy in class_accuracies.items():
        print(f"{class_name}: {accuracy:.4f}")

    avg_class_accuracy = sum(class_accuracies.values()) / len(class_accuracies)
    print(f"\nAverage Class Accuracy: {avg_class_accuracy:.4f}")