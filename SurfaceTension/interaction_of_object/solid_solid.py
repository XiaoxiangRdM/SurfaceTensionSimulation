import numpy as np
import trimesh

class SolidSolidInteraction():
    def __init__(self, solid1, solid2):
        self.solid1 = solid1
        self.solid2 = solid2
        # 弹性系数
        self.ela_coeff = 1.0

    def update(self, time):
        # 计算作用力和力矩
        force = np.zeros(3)
        torque = np.zeros(3)
        # 计算碰撞点
        bounce_point = np.zeros(3)
        normal = np.zeros(3)
        r_1 = bounce_point - self.solid1.centroid
        r_2 = bounce_point - self.solid2.centroid
        delta_v = self.solid1.velo + np.cross(self.solid1.ang_velo, r_1) - self.solid2.velo - np.cross(self.solid2.ang_velo, r_2)
        coeff = (self.solid1.mass + self.solid2.mass) / (self.solid1.mass * self.solid2.mass) + np.dot(normal, np.cross((np.linalg.inv(self.solid1.iner) @ np.cross(r_1, normal)), r_1) + np.cross((np.linalg.inv(self.solid2.iner) @ np.cross(r_2, normal)), r_2))
        p = ((1 + self.ela_coeff) * np.dot(normal, delta_v)) * np.linalg.inv(coeff) @ normal
        # 计算碰撞力和碰撞力矩
        # 更新 solid1
        # 更新 solid2

    def check_collision(self):
        # 计算两个物体的最短距离
        # 如果最短距离小于阈值，返回 True
        # 否则返回 False