import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0, 1], [-1, -0.5]])
B = np.array([[0], [1]])
S = np.identity(2)
Q = np.identity(2)
R = 10 * np.identity(1)

control_dt = 0.1
duration = 10
N = int(duration / control_dt)

u_init = np.array([[2]])
x_init = np.array([[1], [0]])
x = [x_init]
u = []
P = [None] * N
K = [None] * N
P[0] = S
K[0] = np.zeros((1, 2))

A = control_dt * A + np.identity(2)
B = control_dt * B

# Calculate the LQR gain
for k in range(1, N):
    K[k] = np.linalg.inv(B.T @ P[k - 1] @ B + R) @ B.T @ P[k - 1] @ A
    P[k] = (
        np.transpose(A - B @ K[k]) @ P[k - 1] @ (A - B @ K[k])
        + Q
        + np.transpose(K[k]) @ R @ K[k]
    )


for k in range(N):
    x_next = A @ x[k] + B @ (-K[N - k - 1] @ x[k])
    x.append(x_next)
    u.append(-K[N - k - 1] @ x[k])


x_array = np.array(x).squeeze()
u_array = np.array(u).squeeze()
time = np.arange(0, duration + control_dt, control_dt)
print(len(time))
print(len(x_array[:, 0]))
print(K[N - 1])
# for i in range(10 + 1):
#     print(i)
# a = np.arange(0, 1 + 0.1, 0.1)
# print(a)

# Plotting the results
plt.figure(figsize=(10, 6))

plt.subplot(211)
plt.plot(time, x_array[:, 0], label="$x$ (Position)")
plt.plot(time, x_array[:, 1], label="$\dot x$ (Velocity)")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.title("State Trajectory (Free Response)")
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(u_array, label="$u$ (Control Input)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
K = np.array(K).squeeze()
print(K[:, 0])
print(K)
plt.plot(K[:, 0])

plt.show()
