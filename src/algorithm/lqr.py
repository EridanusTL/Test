import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import (
    schur,
    ordqz,
    solve,
    norm,
    inv,
    expm,
    solve_continuous_are,
    solve_discrete_are,
)


def SolveDARE(A, B, Q, R, Sa):
    P = Sa
    epsilon = 1e-3
    error = 1e6
    P_pre = np.zeros((A.shape[0], A.shape[0]))
    K_pre = np.zeros((R.shape[0], A.shape[0]))
    index = 0
    while error > epsilon:
        K = np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A
        # P = np.transpose(A - B @ K) @ P @ (A - B @ K) + Q + np.transpose(K) @ R @ K
        P = A.T @ P @ A - A.T @ P @ B @ K + Q
        error = np.abs(np.max(K_pre - K))
        K_pre = K
        index += 1

    # if index == 10000:
    #     print("DARE did not converge")
    print("DARE converged in ", index, " iterations")
    K = np.linalg.inv(B.T @ P @ B + R) @ B.T @ P @ A

    return K


A = np.array([[0, 1], [-1, -0.5]])
B = np.array([[0], [1]])
Q = np.diag([1, 1])
R = 0.1 * np.identity(1)
S = np.identity(2)
control_dt = 0.001
duration = 20
N = int(duration / control_dt)
u = np.array([[0]])
xd = np.array([[0], [0.2]])
x_init = np.array([[0], [0]])
x_history = [x_init]
u_history = []
xd_history = [xd]

Ad = np.array([[0, 1], [0, 0]])
Aa = np.zeros((5, 5))
Ba = np.zeros((5, 1))
Ca = np.zeros((2, 5))

# Calculate the LQR gain
# Discrete
A = control_dt * A + np.identity(2)
B = control_dt * B
Ad = control_dt * Ad + np.identity(2)

# A_disc = expm(A * control_dt)
# B_disc = inv(A) @ (A_disc - np.eye(2)) @ B
# Ad_disc = expm(Ad * control_dt)
# A = A_disc
# B = B_disc
# Ad = Ad_disc


Aa[0:2, 0:2] = A
Aa[0:2, 4:5] = B
Aa[2:4, 2:4] = Ad
Aa[4, 4] = 1
Ba[0:2, :] = B
Ba[4] = 1
Ca[0:2, 0:2] = np.identity(2)
Ca[0:2, 2:4] = -np.identity(2)
Qa = Ca.T @ Q @ Ca
Sa = Ca.T @ S @ Ca
xa = np.vstack([x_init, xd, np.array([[0]])])
xa_history = [xa]
print("Aa\n", Aa)
print("Ba\n", Ba)
print("Qa\n", Qa)
print("Sa\n", Sa)
print("R\n", R)

Ka = SolveDARE(Aa, Ba, Qa, R, Sa)
print("Ka\n", Ka)
# Ka = np.array([[1.48250088, 2.36127346, -2.05767269, -2.69111179, 0.57517181]])
x = x_init
for k in range(N):
    if k == 5 / control_dt:
        xd = np.array([xd[0], [-0.2]])
    elif k == 10 / control_dt:
        xd = np.array([xd[0], [0.2]])
    elif k == 15 / control_dt:
        xd = np.array([xd[0], [-0.2]])

    u_delta = -Ka @ xa
    u = u + u_delta

    x = A @ x + B @ u
    xd = Ad @ xd
    xa = np.vstack([x, xd, u])

    x_history.append(x)
    xd_history.append(xd)
    xa_history.append(xa)
    u_history.append(u)


x_array = np.array(x_history).squeeze()
u_array = np.array(u_history).squeeze()
xd_array = np.array(xd_history).squeeze()

time = np.arange(0, duration + control_dt, control_dt)


# Plotting the results
plt.figure(figsize=(10, 6))

plt.subplot(311)
plt.plot(time, x_array[:, 0], label="$x$ (Position)")
plt.plot(time, xd_array[:, 0], label="$x_{d1}$", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.title("State Trajectory (Free Response)")
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.plot(time, x_array[:, 1], label="$\\dot x$ (Velocity)")
plt.plot(time, xd_array[:, 1], label="$x_{d2}$", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.grid(True)

plt.subplot(313)
plt.plot(u_array, label="$u$ (Control Input)")
plt.legend()
plt.grid(True)

plt.show()
