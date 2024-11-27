import numpy as np
from SurfaceTension.objects.solid import BasicSolidObject as BasicSolidObject

class Force:
    """Base class for all forces"""
    def __init__(self, name, object, dt=3e-4):
        self.name = name
        self.object = object
        self.dt = dt
        
    def force_and_torque(self):
        pass
    
class Rope(Force):
    def __init__(self, name, object, length, stationary_point, hanging_point, dt):
        super().__init__(name, object, dt)
        """
        Light rope, attached to the environment and the object. No elasticity. No mass.
        length: rope length
        stationary_point: point where the rope is attached to the environment, under ground coordinate system
        hanging_point: point where the rope is attached to the object, under the object's coordinate system
        """
        self.length = length
        # self.stationary_point = stationary_point
        self.hanging_point = self.object.centroid + self.object.rotation_matrix @ hanging_point
        self.arm_of_force = stationary_point - self.object.centroid
        
        self.r_relative = self.hanging_point - stationary_point
        self.distance = np.linalg.norm(self.r_relative)
        self.r_hat = self.r_relative / self.distance
        
        self.velo = self.object.velo + np.cross(self.object.ang_velo, self.hanging_point - self.object.centroid)

        
    def get_force_and_torque(self):
        v_r = np.dot(self.velo, self.r_hat) # relative velocity along the rope
        mass = self.object.mass
        K = 1e3 * mass
        beta = 2 * np.sqrt(K * mass) # damping coefficient
        force = - (K * (self.distance - self.length) + beta * v_r) * self.r_hat
        return force, np.cross(self.arm_of_force, force)
 
