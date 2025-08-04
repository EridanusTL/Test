import numpy as np
import matplotlib.pyplot as plt


def cubic_hermite_interpolation(x0, x1, f0, f1, df0, df1, x):
    """
    三次Hermite插值（两点，带速度约束）

    参数:
    x0, x1: 两个节点
    f0, f1: 节点处的函数值
    df0, df1: 节点处的导数值（速度）
    x: 要计算插值的点

    返回:
    插值结果
    """
    # 标准化到[0,1]区间
    t = (x - x0) / (x1 - x0)
    h = x1 - x0

    # 基函数
    h0 = 2 * t**3 - 3 * t**2 + 1
    h1 = -2 * t**3 + 3 * t**2
    h2 = t**3 - 2 * t**2 + t
    h3 = t**3 - t**2

    # 插值结果
    result = f0 * h0 + f1 * h1 + h * df0 * h2 + h * df1 * h3
    return result


# 示例：轨迹规划
# 机器人从位置0以速度1开始，到位置10以速度-0.5结束
x0, x1 = 0, 5  # 时间区间
pos0, pos1 = 0, 10  # 位置
vel0, vel1 = 1, -0.5  # 速度

# 生成轨迹
t = np.linspace(x0, x1, 100)
position = np.array(
    [cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, ti) for ti in t]
)

# 计算速度（导数）
dt = t[1] - t[0]
velocity = np.gradient(position, dt)

# 绘制结果
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 位置曲线
ax1.plot(t, position, "b-", linewidth=2, label="位置轨迹")
ax1.plot([x0, x1], [pos0, pos1], "ro", markersize=8, label="端点位置")
ax1.set_xlabel("时间")
ax1.set_ylabel("位置")
ax1.set_title("三次Hermite插值：位置轨迹")
ax1.grid(True)
ax1.legend()

# 速度曲线
ax2.plot(t, velocity, "r-", linewidth=2, label="速度轨迹")
ax2.plot([x0, x1], [vel0, vel1], "go", markersize=8, label="端点速度约束")
ax2.set_xlabel("时间")
ax2.set_ylabel("速度")
ax2.set_title("速度变化（导数）")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

print(f"起点: 位置={pos0}, 速度={vel0}")
print(f"终点: 位置={pos1}, 速度={vel1}")
print(f"实际终点速度: {velocity[-1]:.3f}")

# 验证约束条件
print("\n验证约束条件:")
print(
    f"H({x0}) = {cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x0):.6f}, 应该等于 {pos0}"
)
print(
    f"H({x1}) = {cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x1):.6f}, 应该等于 {pos1}"
)

# 数值验证导数
eps = 1e-6
df_x0 = (
    cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x0 + eps)
    - cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x0 - eps)
) / (2 * eps)
df_x1 = (
    cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x1 + eps)
    - cubic_hermite_interpolation(x0, x1, pos0, pos1, vel0, vel1, x1 - eps)
) / (2 * eps)

print(f"H'({x0}) ≈ {df_x0:.6f}, 应该等于 {vel0}")
print(f"H'({x1}) ≈ {df_x1:.6f}, 应该等于 {vel1}")
