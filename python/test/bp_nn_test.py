import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = ["Microsoft YaHei"]


# 激活函数和导数
def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 防止溢出


def sigmoid_derivative(x):
    """Sigmoid函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)


class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1):
        """
        简单的两层神经网络
        input_size: 输入层神经元数量
        hidden_size: 隐藏层神经元数量
        output_size: 输出层神经元数量
        """
        self.learning_rate = learning_rate

        # 随机初始化权重和偏置
        # 输入层到隐藏层的权重 (input_size × hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))

        # 隐藏层到输出层的权重 (hidden_size × output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # 用于存储中间计算结果
        self.z1 = None  # 隐藏层的线性输出
        self.a1 = None  # 隐藏层的激活输出
        self.z2 = None  # 输出层的线性输出
        self.a2 = None  # 输出层的激活输出

        # 记录训练过程
        self.loss_history = []

    def forward(self, X):
        """前向传播"""
        print(f"前向传播过程:")
        print(f"输入 X shape: {X.shape}")
        print(f"X:\n{X}")

        # 输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        print(f"\n隐藏层:")
        print(f"z1 = X·W1 + b1:")
        print(f"z1 shape: {self.z1.shape}")
        print(f"a1 = sigmoid(z1):")
        print(f"a1 shape: {self.a1.shape}")

        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        print(f"\n输出层:")
        print(f"z2 = a1·W2 + b2:")
        print(f"z2 shape: {self.z2.shape}")
        print(f"a2 = sigmoid(z2):")
        print(f"a2 shape: {self.a2.shape}")
        print(f"最终输出: {self.a2.flatten()}")

        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]  # 样本数量

        print(f"\n反向传播过程:")
        print(f"真实标签 y: {y.flatten()}")
        print(f"预测输出 output: {output.flatten()}")

        # 输出层的误差
        # dL/da2 = 2*(a2 - y)  (均方误差的导数)
        dL_da2 = 2 * (output - y) / m
        print(f"\n输出层梯度 dL/da2: {dL_da2.flatten()}")

        # 输出层参数的梯度
        # dL/dz2 = dL/da2 * da2/dz2 = dL/da2 * sigmoid'(z2)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)
        print(f"dL/dz2: {dL_dz2.flatten()}")

        # dL/dW2 = dL/dz2 * dz2/dW2 = a1^T · dL/dz2
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        print(f"dL/dW2 shape: {dL_dW2.shape}")

        # dL/db2 = dL/dz2 (对偏置的梯度)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        print(f"dL/db2: {dL_db2.flatten()}")

        # 隐藏层的误差（链式法则）
        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 · W2^T
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        print(f"\n隐藏层梯度 dL/da1 shape: {dL_da1.shape}")

        # 隐藏层参数的梯度
        # dL/dz1 = dL/da1 * da1/dz1 = dL/da1 * sigmoid'(z1)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        print(f"dL/dz1 shape: {dL_dz1.shape}")

        # dL/dW1 = dL/dz1 * dz1/dW1 = X^T · dL/dz1
        dL_dW1 = np.dot(X.T, dL_dz1)
        print(f"dL/dW1 shape: {dL_dW1.shape}")

        # dL/db1 = dL/dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        print(f"dL/db1 shape: {dL_db1.shape}")

        # 更新参数
        print(f"\n参数更新:")
        print(f"学习率: {self.learning_rate}")

        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1

        print(f"参数更新完成")

    def train_step(self, X, y):
        """训练一步"""
        # 前向传播
        output = self.forward(X)

        # 计算损失
        loss = np.mean((output - y) ** 2)
        self.loss_history.append(loss)

        # 反向传播
        self.backward(X, y, output)

        return loss

    def train(self, X, y, epochs=1000, verbose_step=100):
        """训练网络"""
        print(f"开始训练，共 {epochs} 轮")
        print("=" * 60)

        for epoch in range(epochs):
            loss = self.train_step(X, y)

            if epoch % verbose_step == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                if epoch < 200:  # 前面几轮显示详细过程
                    print("=" * 60)

        print(f"训练完成！最终损失: {self.loss_history[-1]:.6f}")

    def predict(self, X):
        """预测"""
        return self.forward(X)


# 准备XOR数据
def create_xor_data():
    """创建XOR数据集"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y


# 可视化函数
def visualize_results(network, X, y):
    """可视化训练结果"""
    plt.figure(figsize=(15, 5))

    # 损失函数变化
    plt.subplot(1, 3, 1)
    plt.plot(network.loss_history)
    plt.title("训练损失变化")
    plt.xlabel("迭代次数")
    plt.ylabel("损失值")
    plt.yscale("log")
    plt.grid(True)

    # 预测结果
    plt.subplot(1, 3, 2)
    predictions = network.predict(X)

    # 绘制真实值和预测值
    x_pos = np.arange(len(X))
    width = 0.35

    plt.bar(x_pos - width / 2, y.flatten(), width, label="真实值", alpha=0.7)
    plt.bar(x_pos + width / 2, predictions.flatten(), width, label="预测值", alpha=0.7)

    plt.xlabel("样本")
    plt.ylabel("输出值")
    plt.title("XOR预测结果")
    plt.xticks(x_pos, ["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 决策边界
    plt.subplot(1, 3, 3)

    # 创建网格
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = network.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap="RdYlBu")
    plt.colorbar(label="预测值")

    # 绘制数据点
    colors = ["blue", "red"]
    for i in range(len(X)):
        plt.scatter(
            X[i, 0], X[i, 1], c=colors[int(y[i])], s=100, edgecolor="black", linewidth=2
        )
        plt.annotate(
            f"({X[i,0]},{X[i,1]})",
            (X[i, 0], X[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("输入1")
    plt.ylabel("输入2")
    plt.title("XOR决策边界")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 创建数据
    X, y = create_xor_data()

    print("XOR问题数据:")
    print("输入 X:")
    print(X)
    print("输出 y:")
    print(y.flatten())
    print()

    # 创建网络
    network = SimpleNeuralNetwork(
        input_size=2, hidden_size=4, output_size=1, learning_rate=0.5
    )

    # 显示初始参数
    print("初始参数:")
    print(f"W1 shape: {network.W1.shape}")
    print(f"W2 shape: {network.W2.shape}")
    print()

    # 训练网络（只显示前几轮的详细过程）
    print("训练过程（显示前2轮详细步骤）:")
    print("=" * 60)

    # 详细显示前2轮
    for epoch in range(2):
        print(f"\n第 {epoch + 1} 轮训练:")
        print("-" * 40)
        loss = network.train_step(X, y)
        print(f"本轮损失: {loss:.6f}")
        print("=" * 60)

    # 继续训练但不显示详细过程
    print("\n继续训练...")
    for epoch in range(2, 2000):
        network.train_step(X, y)
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {network.loss_history[-1]:.6f}")

    # 最终测试
    print(f"\n最终测试结果:")
    print("=" * 40)
    final_predictions = network.predict(X)

    print("输入 -> 真实值 -> 预测值 -> 误差")
    for i in range(len(X)):
        error = abs(y[i, 0] - final_predictions[i, 0])
        print(f"{X[i]} -> {y[i, 0]} -> {final_predictions[i, 0]:.4f} -> {error:.4f}")

    # 可视化结果
    visualize_results(network, X, y)
