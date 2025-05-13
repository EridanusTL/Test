import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Ubuntu Sans"  # 替换为你选择的字体


# 定义摆线的参数方程（时间参数化，确保起止速度为0）
def cycloid_motion(t, T, start_point, height, end_point):
    """生成摆线轨迹，t为时间数组，T为总时间"""
    # 计算摆线几何参数
    distance_AC = np.linalg.norm(end_point - start_point)
    r = distance_AC / (2 * np.pi)  # 标准摆线半径
    scale_y = (height - start_point[1]) / (2 * r)  # 垂直缩放因子

    # 时间参数化：θ(t)从0到2π，且起止点导数为0
    theta = np.pi * (1 - np.cos(np.pi * t / T))  # θ(t) 的表达式
    dtheta_dt = (np.pi**2 / T) * np.sin(np.pi * t / T)  # dθ/dt

    # 标准摆线坐标
    x_std = r * (theta - np.sin(theta))
    y_std = r * (1 - np.cos(theta))

    # 仿射变换到实际轨迹
    x = start_point[0] + (end_point[0] - start_point[0]) / (2 * np.pi * r) * x_std
    y = start_point[1] + scale_y * y_std

    # 计算速度
    vx = (end_point[0] - start_point[0]) / (2 * np.pi) * dtheta_dt * (1 - np.cos(theta))
    vy = (height - start_point[1]) / 2 * np.sin(theta) * dtheta_dt
    speed = np.sqrt(vx**2 + vy**2)

    return x, y, speed, vx, vy


def cycloid_motion2(t, T, start_point, height, end_point):
    theta = t / T * 2 * np.pi
    x = start_point[0] + (end_point[0] - start_point[0]) * (theta - np.sin(theta)) / (
        2 * np.pi
    )
    z = start_point[1] + (height - start_point[1]) * (1 - np.cos(theta)) / 2

    return x, z


# 设置控制点和总时间
start_point = np.array([0, 0])  # 起始点
height = 0.2  # 中间点
end_point = np.array([0.4, 0])  # 终止点
T = 0.4  # 总时间

# 生成时间序列和轨迹
t = np.linspace(0, T, 500)
x, y, speed, vx, vy = cycloid_motion(t, T, start_point, height, end_point)
xx, zz = cycloid_motion2(t, T, start_point, height, end_point)

# 绘制轨迹和速度曲线
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# 轨迹图
ax1.plot(xx, zz, label="cycloidal path")
ax1.scatter(
    [start_point[0], end_point[0]],
    [start_point[1], end_point[1]],
    color="red",
    label="control point",
)
ax1.set_title("Cycloidal Path")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.grid(True)

# 速度曲线
ax2.plot(t, vy, color="orange", label="velocity")
ax2.set_title("Velocity Curve")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("velocity")
ax2.legend()
ax2.grid(True)

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
ax3.plot(t, x)
ax3.grid(True)

ax4.plot(t, y)
ax4.grid(True)

plt.tight_layout()
plt.show()
