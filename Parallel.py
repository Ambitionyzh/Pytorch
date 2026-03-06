import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 类的定义和工具函数，放在最外面（老板和工人都要看）
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 🛑 核心修复：把所有的业务逻辑包在这个判断里！
# ==========================================
if __name__ == '__main__':

    # 从这里开始，所有的代码都要往右缩进 4 个空格！

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 开启 num_workers=2
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()

    model = MNIST_Network()

    if gpu_count > 1:
        print(f"🚀 侦测到 {gpu_count} 张显卡！正在启动 DataParallel...")
        model = nn.DataParallel(model)
    else:
        print(f"🐢 目前使用单卡或 CPU 训练: {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 你的报错就是在这里触发的，现在被锁在 if 里面，工人就不会再执行这句了
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], 训练集 Loss: {running_loss / len(train_loader):.4f}")

    print("✅ 多卡并行代码执行完毕！")