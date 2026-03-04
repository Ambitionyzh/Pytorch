import torch
import torch.nn as nn
import torch.nn.functional as F

#下面是一个专门用来识别 MNIST 手写数字（0-9）的经典多层感知机（MLP）的代码骨架
# 1. 定义我们自己的网络类，必须继承 nn.Module
class MNIST_Network(nn.Module):

    # 2. 初始化函数：用来“准备零件”
    def __init__(self):
        super(MNIST_Network, self).__init__()  # 必须写的家规：呼叫父类的初始化

        # 第一层 (全连接层)：输入是 784，输出是我们设定的隐藏层 128
        self.fc1 = nn.Linear(28 * 28, 128)

        # 第二层 (输出层)：输入是上一层的 128，输出必须是 10 (代表 0-9 这十个数字)
        self.fc2 = nn.Linear(128, 10)

    # 3. 前向传播函数：用来“画组装图纸”
    def forward(self, x):
        # 步骤 A：把 2D 的图片 (28x28) 拍平成 1D 的长条 (784)
        x = x.view(-1, 28 * 28)

        # 步骤 B：数据穿过第一层，并套上 ReLU 激活函数
        x = F.relu(self.fc1(x))

        # 步骤 C：数据穿过第二层 (输出层)，得到最终的 10 个打分
        x = self.fc2(x)

        return x


# 4. 实例化我们的模型
model = MNIST_Network()
# 1. 简单粗暴地计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"这个微型大脑的总参数量是: {total_params} 个")

# 2. 详细拆解：看看每层具体有多少个参数
print("\n--- 参数详细账单 ---")
for name, param in model.named_parameters():
    print(f"零件名称: {name.ljust(12)} | 形状: {str(list(param.shape)).ljust(12)} | 参数个数: {param.numel()}")
print(model)