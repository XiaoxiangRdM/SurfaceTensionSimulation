import numpy as np
import matplotlib.pyplot as plt

import utils

# 设置网格大小
NX = 100
NY = 100
NZ = 100

# 创建网格
u = np.zeros((NX, NY, NZ))  # 流体速度定义在面中心
p = np.zeros((NX, NY, NZ))    # 流体压力定义在单元中心

# 模拟一些流体运动（示例）
u[:, :, :] = 1.0



# 更新速度场
p = utils.pressure_solve(u, p, NX, NY, NZ)
u_p = utils.grad(p, NX, NY, NZ)
u_new = u - u_p

# 可视化速度场
X, Y = np.meshgrid(np.arange(NY), np.arange(NX))
U = u_new[1:, :, :]  # x方向速度
V = u_new[:, 1:, :]  # y方向速度
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U[:, :, 50], V[:, :, 50])  # 取Z中间层的速度进行可视化
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity Field')
plt.show()