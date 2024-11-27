import numpy as np
import trimesh
import SurfaceTension.utils as utils

class BasicSolidObject():
    def __init__(self, name, mass, volume, iner, centroid, axis, angle, velo, ang_velo):
        """
        Initialize the BaiscSolidObject.

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
        - is_first_frame (bool): Is the first frame or not. 
        - centroid_last_frame (np.ndarray): Position of the centroid of last frame.
        - rotation_matrix_last_frame (np.ndarray): Rotation matrix of last frame. 
        - velo_last_frame (np.ndarray): Linear velocity of last frame.
        - ang_velo_last_frame (np.ndarray): Angular velocity of last frame.
        - acc_last_frame (np.ndarray): Linear acceleration of last frame.
        - ang_acc_last_frame (np.ndarray): Angular acceleration of last frame.

        Note:
        - The inertia tensor is assumed to be a diagonal matrix for computational efficiency, 
            (Guaranteed by physical law. Under inertia axis coordinate system)
            and computed during initialization.
        - However, most time we use inertia tensor under the ground coordinates system. 
        """
        assert mass > 0, "Mass must be a float larger than zero."
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
        
        self.is_first_frame = True
        self.centroid_last_frame = np.array([0,0,0])
        self.rotation_matrix_last_frame = np.eye(3)
        self.velo_last_frame = np.array([0,0,0])
        self.ang_velo_last_frame = np.array([0,0,0])
        self.acc_last_frame = np.array([0,0,0])
        self.ang_acc_last_frame = np.array([0,0,0])
        
        self.energy = 0
    
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
        axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
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

    def update_kinetic_energy(self): 
        self.energy = 0.5 * (self.mass * self.velo.T @ self.velo + self.ang_velo.T @ self.compute_inertia_ground() @ self.ang_velo)
        
    def update_last_frame_vars(self, acc, ang_acc): 
        self.centroid_last_frame = self.centroid
        self.rotation_matrix_last_frame = self.rotation_matrix
        self.velo_last_frame = self.velo
        self.ang_velo_last_frame = self.ang_velo
        self.acc_last_frame = acc
        self.ang_acc_last_frame = ang_acc

    def update(self, time, force, torque):
        # Detect if is the first frame, if so then initialize 
        
        # Calculate linear acceleration
        lin_acc = force / self.mass
        
        # Calculate inertia tensor under the ground coordinates system
        iner_ground = self.compute_inertia_ground()
        iner_inv_ground = self.compute_inertia_inverse_ground()
        
        # Calsulate skew matrix of angular velocity
        ang_velo_skew = utils.skew_matrix(self.ang_velo)
        
        # Calculate angular accelaration
        ang_acc = iner_inv_ground @ (torque - ang_velo_skew @ iner_ground @ self.ang_velo)

        # Check if this is first frame. 
        # If so, initialize linear acceleration and angular accelaration of last frame. 
        if (self.is_first_frame): 
            self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)

        # Update linear motion
        posi_change = self.velo * time + (4 * lin_acc - self.acc_last_frame) * (time ** 2) / 3
        self.centroid += posi_change
        # self.centroid = (2 * self.centroid - self.centroid_last_frame) + self.velo * time + lin_acc * (time ** 2)
        
        # Update linear velocity
        self.velo += (1.5 * lin_acc - 0.5 * self.acc_last_frame) * time
        # self.velo = (self.centroid - self.centroid_last_frame) / (time * 2)
        
        # Update rotation matrix 
        eff_ang_velo = self.ang_velo + \
            (0.5 * np.cross(ang_acc, self.ang_velo) * time + (4 * ang_acc - self.ang_acc_last_frame)) * time / 6
        eff_ang_velo_value = np.linalg.norm(eff_ang_velo)
        if eff_ang_velo_value != 0:
            current_axis = (eff_ang_velo) / eff_ang_velo_value
            self.rotation_matrix = self.compute_rotation_matrix(current_axis, eff_ang_velo_value * time) @ \
                self.rotation_matrix

        # Update angular velocity
        self.ang_velo += (1.5 * ang_acc - 0.5 * self.ang_acc_last_frame) * time
        
        # Update last-frame variables
        self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)
        
        # Update energy 
        self.update_kinetic_energy()
        
        self.is_first_frame = False
        


    def check_inside(self, pos):
        raise NotImplementedError

class SolidObject(BasicSolidObject):
    # Mesh-based solid
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
        - is_first_frame (bool): Is the first frame or not. 
        - centroid_last_frame (np.ndarray): Position of the centroid of last frame.
        - rotation_matrix_last_frame (np.ndarray): Rotation matrix of last frame. 
        - velo_last_frame (np.ndarray): Linear velocity of last frame.
        - ang_velo_last_frame (np.ndarray): Angular velocity of last frame.
        - acc_last_frame (np.ndarray): Linear acceleration of last frame.
        - ang_acc_last_frame (np.ndarray): Angular acceleration of last frame.

        Note:
        - The inertia tensor is assumed to be a diagonal matrix for computational efficiency, 
            (Guaranteed by physical law. Under inertia axis coordinate system)
            and computed during initialization.
        - However, most time we use inertia tensor under the ground coordinates system. 
        
        assert mass > 0, "Mass must be a float larger than zero."
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
        self.ang_velo = ang_velo"""
        super().__init__(name, mass, volume, iner, centroid, axis, angle, velo, ang_velo)
        
        self.obj_file = obj_file
        self.mesh = trimesh.load(obj_file)  # Load .obj file
        
        """self.is_first_frame = True
        self.centroid_last_frame = np.array([0,0,0])
        self.rotation_matrix_last_frame = np.eye(3)
        self.velo_last_frame = np.array([0,0,0])
        self.ang_velo_last_frame = np.array([0,0,0])
        self.acc_last_frame = np.array([0,0,0])
        self.ang_acc_last_frame = np.array([0,0,0])
        
        self.energy = 0
    
        self.axis = axis # stop maintenance
        self.angle = angle # stop maintenance
        
    def compute_rotation_matrix(self, axis, angle):
        ""
        Compute the rotation matrix given an axis and an angle.

        Parameters:
        - axis (np.ndarray): Rotation axis.
        - angle (float): Rotation angle in radians.

        Returns:
        - np.ndarray: 3x3 rotation matrix.
        ""
        axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
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

    def update_kinetic_energy(self): 
        self.energy = 0.5 * (self.mass * self.velo.T @ self.velo + self.ang_velo.T @ self.compute_inertia_ground() @ self.ang_velo)
        
    def update_last_frame_vars(self, acc, ang_acc): 
        self.centroid_last_frame = self.centroid
        self.rotation_matrix_last_frame = self.rotation_matrix
        self.velo_last_frame = self.velo
        self.ang_velo_last_frame = self.ang_velo
        self.acc_last_frame = acc
        self.ang_acc_last_frame = ang_acc

    def update(self, time, force, torque):
        # Detect if is the first frame, if so then initialize 
        
        # Calculate linear acceleration
        lin_acc = force / self.mass
        
        # Calculate inertia tensor under the ground coordinates system
        iner_ground = self.compute_inertia_ground()
        iner_inv_ground = self.compute_inertia_inverse_ground()
        
        # Calsulate skew matrix of angular velocity
        ang_velo_skew = utils.skew_matrix(self.ang_velo)
        
        # Calculate angular accelaration
        ang_acc = iner_inv_ground @ (torque - ang_velo_skew @ iner_ground @ self.ang_velo)

        # Check if this is first frame. 
        # If so, initialize linear acceleration and angular accelaration of last frame. 
        if (self.is_first_frame): 
            self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)

        # Update linear motion
        posi_change = self.velo * time + (4 * lin_acc - self.acc_last_frame) * (time ** 2) / 3
        self.centroid += posi_change
        # self.centroid = (2 * self.centroid - self.centroid_last_frame) + self.velo * time + lin_acc * (time ** 2)
        
        # Update linear velocity
        self.velo += (1.5 * lin_acc - 0.5 * self.acc_last_frame) * time
        # self.velo = (self.centroid - self.centroid_last_frame) / (time * 2)
        
        # Update rotation matrix 
        eff_ang_velo = self.ang_velo + \
            (0.5 * np.cross(ang_acc, self.ang_velo) * time + (4 * ang_acc - self.ang_acc_last_frame)) * time / 6
        eff_ang_velo_value = np.linalg.norm(eff_ang_velo)
        if eff_ang_velo_value != 0:
            current_axis = (eff_ang_velo) / eff_ang_velo_value
            self.rotation_matrix = self.compute_rotation_matrix(current_axis, eff_ang_velo_value * time) @ \
                self.rotation_matrix

        # Update angular velocity
        self.ang_velo += (1.5 * ang_acc - 0.5 * self.ang_acc_last_frame) * time
        
        # Update last-frame variables
        self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)
        
        # Update energy 
        self.update_kinetic_energy()
        
        self.is_first_frame = False"""
        


    def check_inside(self, pos):
        
        ray_direction = [0, 0, 1] # Doesn't matter

        ray_origin = pos
        intersections = self.mesh.ray.intersects_location(ray_origin, ray_direction)
        
        # Detect the number of intersections
        return len(intersections) % 2 == 1

    def get_mesh_data(self):
        """
        Retrieves the vertices and faces data of the mesh.

        Returns:
        - vertices (np.ndarray): Array of vertex coordinates with shape (N, 3).
        - faces (np.ndarray): Array of face vertex indices with shape (M, 3).
        """
        vertices = (self.rotation_matrix @ self.mesh.vertices.T).T + self.centroid  # Get vertex coordinates
        faces = self.mesh.faces  # Get face indices
        return vertices, faces

        
class BallObject(BasicSolidObject): 
    def __init__(self, name, mass, volume, centroid, axis, angle, velo, ang_velo, radius):
        """
        Initialize the BallObject.

        Parameters:
        - name (str): Name of the object.
        - mass (float): Mass of the object.
        - volume (float): Volume of the object.
        - centroid (np.ndarray): Position of the centroid.
        - axis (np.ndarray): Rotation axis.
        - angle (float): Rotation angle in radians.
        - velo (np.ndarray): Linear velocity.
        - ang_velo (np.ndarray): Angular velocity.
        - radius (float): Radius of the ball.
        - is_first_frame (bool): Is the first frame or not. 
        - centroid_last_frame (np.ndarray): Position of the centroid of last frame.
        - rotation_matrix_last_frame (np.ndarray): Rotation matrix of last frame. 
        - velo_last_frame (np.ndarray): Linear velocity of last frame.
        - ang_velo_last_frame (np.ndarray): Angular velocity of last frame.
        - acc_last_frame (np.ndarray): Linear acceleration of last frame.
        - ang_acc_last_frame (np.ndarray): Angular acceleration of last frame.

        Note:
        - The inertia tensor is assumed to be a diagonal matrix for computational efficiency, 
            (Guaranteed by physical law. Under inertia axis coordinate system)
            and computed during initialization.
        - However, most time we use inertia tensor under the ground coordinates system. 
        
        assert mass > 0, "Mass must be a float larger than zero."
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
        self.ang_velo = ang_velo"""
        # Inertia of a ball with fixed mass density is 2/5*m*r^2
        iner = np.full(3, 0.4 * mass * radius**2)
        super().__init__(name, mass, volume, iner, centroid, axis, angle, velo, ang_velo)
        self.radius = radius
        
        """self.is_first_frame = True
        self.centroid_last_frame = np.array([0,0,0])
        self.rotation_matrix_last_frame = np.eye(3)
        self.velo_last_frame = np.array([0,0,0])
        self.ang_velo_last_frame = np.array([0,0,0])
        self.acc_last_frame = np.array([0,0,0])
        self.ang_acc_last_frame = np.array([0,0,0])
        
        self.energy = 0
    
        self.axis = axis # stop maintenance
        self.angle = angle # stop maintenance
        
    def compute_rotation_matrix(self, axis, angle):
        ""
        Compute the rotation matrix given an axis and an angle.

        Parameters:
        - axis (np.ndarray): Rotation axis.
        - angle (float): Rotation angle in radians.

        Returns:
        - np.ndarray: 3x3 rotation matrix.
        ""
        axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
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

    def update_kinetic_energy(self): 
        self.energy = 0.5 * (self.mass * self.velo.T @ self.velo + self.ang_velo.T @ self.compute_inertia_ground() @ self.ang_velo)
        
    def update_last_frame_vars(self, acc, ang_acc): 
        self.centroid_last_frame = self.centroid
        self.rotation_matrix_last_frame = self.rotation_matrix
        self.velo_last_frame = self.velo
        self.ang_velo_last_frame = self.ang_velo
        self.acc_last_frame = acc
        self.ang_acc_last_frame = ang_acc

    def update(self, time, force, torque):
        # Detect if is the first frame, if so then initialize 
        
        # Calculate linear acceleration
        lin_acc = force / self.mass
        
        # Calculate inertia tensor under the ground coordinates system
        iner_ground = self.compute_inertia_ground()
        iner_inv_ground = self.compute_inertia_inverse_ground()
        
        # Calsulate skew matrix of angular velocity
        ang_velo_skew = utils.skew_matrix(self.ang_velo)
        
        # Calculate angular accelaration
        ang_acc = iner_inv_ground @ (torque - ang_velo_skew @ iner_ground @ self.ang_velo)

        # Check if this is first frame. 
        # If so, initialize linear acceleration and angular accelaration of last frame. 
        if (self.is_first_frame): 
            self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)

        # Update linear motion
        posi_change = self.velo * time + (4 * lin_acc - self.acc_last_frame) * (time ** 2) / 3
        self.centroid += posi_change
        # self.centroid = (2 * self.centroid - self.centroid_last_frame) + self.velo * time + lin_acc * (time ** 2)
        
        # Update linear velocity
        self.velo += (1.5 * lin_acc - 0.5 * self.acc_last_frame) * time
        # self.velo = (self.centroid - self.centroid_last_frame) / (time * 2)
        
        # Update rotation matrix 
        eff_ang_velo = self.ang_velo + \
            (0.5 * np.cross(ang_acc, self.ang_velo) * time + (4 * ang_acc - self.ang_acc_last_frame)) * time / 6
        eff_ang_velo_value = np.linalg.norm(eff_ang_velo)
        if eff_ang_velo_value != 0:
            current_axis = (eff_ang_velo) / eff_ang_velo_value
            self.rotation_matrix = self.compute_rotation_matrix(current_axis, eff_ang_velo_value * time) @ \
                self.rotation_matrix

        # Update angular velocity
        self.ang_velo += (1.5 * ang_acc - 0.5 * self.ang_acc_last_frame) * time
        
        # Update last-frame variables
        self.update_last_frame_vars(acc=lin_acc, ang_acc=ang_acc)
        
        # Update energy 
        self.update_kinetic_energy()
        
        self.is_first_frame = False"""
        


    def check_inside(self, pos):
        
        #ray_direction = [0, 0, 1] # Doesn't matter

        #ray_origin = pos
        #intersections = self.mesh.ray.intersects_location(ray_origin, ray_direction)
        
        # Detect the number of intersections
        return np.linalg.norm(pos - self.centroid) <= self.radius

    #def get_mesh_data(self):
        """
        Retrieves the vertices and faces data of the mesh.

        Returns:
        - vertices (np.ndarray): Array of vertex coordinates with shape (N, 3).
        - faces (np.ndarray): Array of face vertex indices with shape (M, 3).
        """
        #vertices = (self.rotation_matrix @ self.mesh.vertices.T).T + self.centroid  # Get vertex coordinates
        #faces = self.mesh.faces  # Get face indices
        #return vertices, faces
    
