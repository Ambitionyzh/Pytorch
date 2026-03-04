import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 准备阶段 (前置工作)
# ==========================================

# 1. 建立"仓库"和调度"车队"
# 这里我们捏造了一点数据：规律显然是 y = 2 * x
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
dataset = TensorDataset(x_data, y_data) # 把 x 和 y 打包进标准仓库
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # 每趟车拉2条数据

# 2. 建立模型
# nn.Linear(1, 1) 代表输入是 1 维，输出是 1 维。
# PyTorch 会在这个盒子里自动帮你生成 w 和 b，并且自动设置 requires_grad=True！
model = nn.Linear(1, 1)

# 3. 定义"尺子"(损失函数) 和 "下山步伐"(优化器)
criterion = nn.MSELoss() # 均方误差 (我们昨天学的)
# SGD 就是随机梯度下降。lr=0.01 就是学习率 α。它接管了模型里的所有参数 (w 和 b)。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# ==========================================
# 训练大循环 (核心骨架)
# ==========================================
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, labels in dataloader: # 物流车送来了一批数据
        
        # 🔥 深度学习训练最核心的 5 步曲 🔥
        
        # 第 1 步：前向传播 (Forward Pass) -> 让模型蒙眼猜一猜
        y_pred = model(inputs)
        
        # 第 2 步：计算误差 (Compute Loss) -> 拿尺子量一量错得有多离谱
        loss = criterion(y_pred, labels)
        
        # 第 3 步：清空旧梯度 (Zero Gradients) -> 填昨天的坑！
        optimizer.zero_grad()
        
        # 第 4 步：反向传播 (Backward Pass) -> PyTorch 魔法：自动算坡度 (梯度)
        loss.backward()
        
        # 第 5 步：更新参数 (Update Weights) -> 迈出一步：w = w - lr * 坡度
        optimizer.step()

    # 每隔 20 轮打印一下进度，看看误差是不是越来越小了
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 验货阶段
# ==========================================
print("\n训练完成！来看看模型自己学到的公式参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.item():.4f}") 
# 理想情况下，weight 会接近 2.0，bias 会接近 0.0