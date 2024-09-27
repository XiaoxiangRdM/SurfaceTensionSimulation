import numpy as np
import matplotlib.pyplot as plt
import utils

class fluid_object():
    def __init__(self, NX, NY, NZ):
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.u = np.zeros((NX+1, NY, NZ))  # 流体速度定义在面中心
        self.p = np.zeros((NX, NY, NZ))    # 流体压力定义在单元中心

    
    def pressure_solve(self):
        """简单的压力求解，使用雅可比迭代"""
        tolerance = 1e-6
        p_new = self.p.copy()
        max_iter = 10000
        for _ in range(max_iter):
            p_old = p_new.copy()
            divsp = self.div()
            for i in range(self.NX):
                for j in range(self.NY):
                    for k in range(self.NZ):
                        sum_p = 0
                        if i > 0:
                            sum_p += p_old[i-1, j, k]
                        if i < self.NX-1:
                            sum_p += p_old[i+1, j, k]
                        if j > 0:
                            sum_p += p_old[i, j-1, k]
                        if j < self.NY-1:
                            sum_p += p_old[i, j+1, k]
                        if k > 0:
                            sum_p += p_old[i, j, k-1]
                        if k < self.NZ-1:
                            sum_p += p_old[i, j, k+1]
                        p_new[i, j, k] = (divsp[i, j, k] + sum_p) / 6
            if np.max(np.abs(p_new - p_old)) < tolerance:
                break
        self.p = p_new

    def update_velocity_field(self):
        """更新速度场"""
        u_p = utils.grad()
        self.u -= u_p

    def run_simulation(self):
        """运行模拟"""
        self.pressure_solve()
        self.update_velocity_field()

    def visualize_velocity_field(self):
        """可视化速度场"""
        X, Y = np.meshgrid(np.arange(self.NY), np.arange(self.NX))
        U = self.u[1:, :, :]  # x方向速度
        V = self.u[:, 1:, :]  # y方向速度
        plt.figure(figsize=(10, 6))
        plt.quiver(X, Y, U[:, :, self.NZ//2], V[:, :, self.NZ//2])  # 取Z中间层的速度进行可视化
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Velocity Field')
        plt.show()

def main():
    NX = 100
    NY = 100
    NZ = 100
    simulation = fluid_object(NX, NY, NZ)
    simulation.u[:, :, :] = 1.0  # 模拟一些流体运动
    simulation.run_simulation()
    simulation.visualize_velocity_field()

if __name__ == "__main__":
    main()