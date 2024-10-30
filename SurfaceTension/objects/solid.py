import numpy as np
import trimesh

class SolidObject():
    def __init__(self, name, mass, volume, iner, centroid, angle, velo, ang_velo, obj_file):
        assert iner.shape == (3, 3), "Inertia matrix must be a 3x3 matrix."
        assert np.allclose(iner, iner.T), "Inertia matrix must be symmetric."
        assert np.all(np.linalg.eigvals(iner) > 0), "Inertia matrix must be positive definite."
        self.name = name
        self.mass = mass
        self.volume = volume
        self.iner = iner # 3x3 positive definite matrix
        self.centroid = centroid # position of centroid 
        self.angle = angle
        self.velo = velo
        self.ang_velo = ang_velo
        self.mesh = trimesh.load(obj_file)  # 加载.obj文件

    def update(self, time, force, torque):
        acc = force / self.mass
        posi_change = self.velo * time + 0.5 * acc * time * time
        self.centroid += posi_change
        self.velo += acc * time
        omega = np.array([[0, -self.ang_velo[2], self.ang_velo[1]], [self.ang_velo[2], 0, -self.ang_velo[0]], [-self.ang_velo[1], self.ang_velo[0], 0]])
        iner_inv = np.linalg.inv(self.iner)
        ang_acc = iner_inv @ torque + ((iner_inv @ omega @ self.iner) + omega) @ self.ang_velo # 使用矩阵乘法
        self.angle += self.ang_velo * time + 0.5 * ang_acc * time * time
        self.ang_velo += ang_acc * time
        # 更新 inertial matrix


    def check_inside(self, pos):
        
        # 判断 pos 是否是质心
        if np.allclose(pos, self.centroid):
            ray_direction = [0, 0, 1]  # 向上沿 z 轴发射
        else:
            # 从 pos 向质心发射射线
            ray_direction = self.centroid - pos
            ray_direction /= np.linalg.norm(ray_direction)  # 单位化射线方向

        # 从 pos 发射射线
        ray_origin = pos
        intersections = self.mesh.ray.intersects_location(ray_origin, ray_direction)
        
        # 判断交点数量
        return len(intersections) % 2 == 1


        
        
    
