import torch
import torch.nn as nn
import time


# ==========================================
# 第一步：搭建一个“肉一点”的测试模型
# （为了让编译器有活干，我们多叠几层，并用上 LLM 最爱的 GELU 激活函数）
# ==========================================
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.layers(x)


# 自动侦测你的设备 (AMD 显卡装了 ROCm 也会被识别为 cuda 运行哦！)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的计算设备: {device}")

model = TestModel().to(device)
# 虚构一个 Batch 的数据 (128条数据，每条512个维度)
x = torch.randn(128, 512).to(device)


# ==========================================
# 第二步：定义极其严谨的“专业掐秒表”函数
# ==========================================
def benchmark(model, x, num_runs=1000, name="模型"):
    print(f"⏳ 正在为 [{name}] 进行热身...")
    # ⚠️ 助教高能防坑 1：必须要“热身 (Warm-up)”！
    # torch.compile 第一次运行的时候，底层 Inductor 正在疯狂写 C++ 代码，所以会特别慢。
    # 我们必须先空跑几次，让它把锅烧热，代码编译完！
    for _ in range(10):
        _ = model(x)

    # ⚠️ 助教高能防坑 2：GPU 同步！
    # GPU 算东西是“异步”的（CPU 发号施令完就不管了）。
    # 如果不加这句强制同步，秒表测出来的只是 CPU 发命令的时间，极其不准！
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"🏃‍♂️ [{name}] 热身完毕，正式起跑！测算 {num_runs} 次...")
    start_time = time.time()

    # 核心循环
    for _ in range(num_runs):
        _ = model(x)

    # 再次强制同步，确保最后一次 GPU 计算彻底收尾，再按停秒表
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = (end_time - start_time) * 1000  # 换算成毫秒
    print(f"✅ [{name}] 总耗时: {total_time:.2f} 毫秒 (平均每次 {total_time / num_runs:.3f} 毫秒)")
    return total_time


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🐢 第一回合：原生 PyTorch (Eager 模式)")
    time_eager = benchmark(model, x, name="原生模式")

    print("\n" + "=" * 50)
    print("🚀 第二回合：注入 torch.compile 魔法！")
    # 核心魔法，就这一行！
    compiled_model = torch.compile(model)
    time_compiled = benchmark(compiled_model, x, name="编译模式")

    print("\n" + "=" * 50)
    print("📊 最终战报")
    if time_compiled < time_eager:
        speedup = (time_eager - time_compiled) / time_eager * 100
        print(f"🎉 结论：算子融合成功！Compile 魔法让速度提升了 {speedup:.2f}%！")
    else:
        print("🤔 咦？没有变快？")
        print("助教解析：如果你用的是 CPU，或者模型太小，编译的开销可能会盖过收益。")
        print("Compile 的真正杀伤力在于复杂的 GPU 算子融合！")