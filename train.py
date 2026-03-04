import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 第一步：准备“数据车队”
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
# 训练集 (教材)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 测试集 (期末考卷)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ==========================================
# 第二步：搭建“AI 大脑”
# ==========================================
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 隐藏层
        self.fc2 = nn.Linear(128, 10)  # 输出层 (10个数字的打分)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 拉平图片
        x = F.relu(self.fc1(x))  # 激活函数
        x = self.fc2(x)
        return x


model = MNIST_Network()

# ==========================================
# 第三步：准备“尺子”和“下山算法”
# ==========================================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失 (分类题专用)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# ==========================================
# 第四步：训练与考试大循环 🔥
# ==========================================
epochs = 5  # 整个教材学 5 遍

for epoch in range(epochs):

    # ------------------ [训练阶段] ------------------
    model.train()  # 告诉模型：现在是学习状态！
    running_loss = 0.0

    for images, labels in train_loader:
        # 核心 5 步曲：
        optimizer.zero_grad()  # 1. 清空旧坡度 (梯度)
        outputs = model(images)  # 2. 前向传播：蒙眼做题
        loss = criterion(outputs, labels)  # 3. 计算误差：核对答案
        loss.backward()  # 4. 反向传播：PyTorch自动算坡度
        optimizer.step()  # 5. 更新权重：迈出下山的一步

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], 训练集平均 Loss: {running_loss / len(train_loader):.4f}")

    # ------------------ [考试阶段] ------------------
    model.eval()  # 告诉模型：现在是考试状态，不要乱动权重！
    correct = 0  # 答对的题数
    total = 0  # 总题数

    # 考试时严禁翻书学习！所以要关闭梯度追踪，省内存又提速
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  # 模型给出 10 个数字的打分

            # 找到得分最高的那个选项，作为最终预测答案
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)  # 累加总题数
            correct += (predicted == labels).sum().item()  # 如果预测和标签一样，答对题数 +1

    print(f"-> 期末考试成绩: 准确率 {100 * correct / total:.2f}%\n")

print("恭喜！AI 训练圆满结束！")
torch.save(model.state_dict(), 'mnist_perfect_brain.pth')

print("✅ 模型参数已成功保存到 mnist_perfect_brain.pth！")