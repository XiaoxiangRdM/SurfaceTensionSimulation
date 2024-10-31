import numpy as np
import trimesh
import SurfaceTension.utils as utils

class SolidObject():
    def __init__(self, name, mass, volume, iner, centroid, axis, angle, velo, ang_velo, obj_file):
        """
        Initialize the SolidObject.

        Parameters:
        - name (str): Name of the object.
        - mass (float): Mass of the object.
        - volume (float): Volume of the object.
        - iner (np.ndarray): Inertia tensor as a 3x3 positive definite matrix.
        - centroid (np.ndarray): Position of the centroid.
        - axis (np.ndarray): Rotation axis.
        - angle (float): Rotation angle in radians.
        - velo (np.ndarray): Linear velocity.
        - ang_velo (np.ndarray): Angular velocity.
        - obj_file (str): Path to the .obj file.

        Note:
        - The inertia tensor is assumed to be a diagonal matrix for computational efficiency, 
            (Guaranteed by physical law. Under inertia axis coordinate system)
            and computed during initialization.
        - However, most time we use inertia tensor under the ground coordinates system. 
        """
        assert len(iner) == 3, "Inertia tensor must be a length-3 array for a diagonal matrix."
        assert all(i > 0 for i in iner), "Each diagonal element of the inertia tensor must be greater than zero."
        self.name = name
        self.mass = mass
        self.volume = volume
        self.iner = np.diag(iner)  # Convert to diagonal matrix
        self.iner_inv = np.diag(1.0 / iner)  # Inverse of the diagonal matrix
        self.centroid = centroid
        self.rotation_matrix = self.compute_rotation_matrix(axis, angle)
        self.velo = velo
        self.ang_velo = ang_velo
        self.obj_file = obj_file
        self.mesh = trimesh.load(obj_file)  # Load .obj file
    
        self.axis = axis # stop maintenance
        self.angle = angle # stop maintenance
        
    def compute_rotation_matrix(self, axis, angle):
        """
        Compute the rotation matrix given an axis and an angle.

        Parameters:
        - axis (np.ndarray): Rotation axis.
        - angle (float): Rotation angle in radians.

        Returns:
        - np.ndarray: 3x3 rotation matrix.
        """
        # axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R
    
    def compute_inertia_ground(self): 
        return self.rotation_matrix @ self.iner @ self.rotation_matrix.T
    
    def compute_inertia_inverse_ground(self): 
        return self.rotation_matrix @ self.iner_inv @ self.rotation_matrix.T

    def update(self, time, force, torque):
        # Update linear motion
        lin_acc = force / self.mass
        posi_change = self.velo * time + 0.5 * lin_acc * time * time
        self.centroid += posi_change
        self.velo += lin_acc * time
        
        # Calculate inertia tensor under the ground coordinates system
        iner_ground = self.compute_inertia_ground()
        iner_inv_ground = self.compute_inertia_inverse_ground()
        
        # Calsulate skew matrix of angular velocity
        ang_velo_skew = utils.skew_matrix(self.ang_velo)
        
        # Calculate angular accelaration
        ang_acc = iner_inv_ground @ (torque - ang_velo_skew @ iner_ground @ self.ang_velo)

        
        # Update rotation matrix 
        avg_ang_velo = self.ang_velo + 0.5 * ang_acc * time
        avg_ang_velo_value = np.linalg.norm(avg_ang_velo)
        if avg_ang_velo_value != 0:
            current_axis = (avg_ang_velo) / avg_ang_velo_value
            self.rotation_matrix = self.rotation_matrix @ self.compute_rotation_matrix(current_axis, avg_ang_velo_value * time)

        
        # Update angular velocity
        self.ang_velo += time * ang_acc



    def check_inside(self, pos):
        
        ray_direction = [0, 0, 1] # doesn't matter

        ray_origin = pos
        intersections = self.mesh.ray.intersects_location(ray_origin, ray_direction)
        
        # 判断交点数量
        return len(intersections) % 2 == 1


        
        
    
