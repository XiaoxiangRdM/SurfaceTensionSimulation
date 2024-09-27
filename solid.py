import numpy as np

class solid_object():
    def __init__(self, name, mass, volume, iner, posi, angle, velo, ang_velo):
        self.name = name
        self.mass = mass
        self.volume = volume
        self.iner = iner
        self.posi = posi
        self.angle = angle
        self.velo = velo
        self.ang_velo = ang_velo
        
    def update(self, time, force, torque):
        acc = force / self.mass
        posi_change = self.velo * time + 0.5 * acc * time * time
        self.posi = self.posi + posi_change
        self.velo = self.velo + acc * time
        ang_acc = torque * np.linalg.inv(self.iner)
        self.angle = self.angle + self.ang_velo * time + 0.5 * ang_acc * time * time
        self.ang_velo = self.ang_velo + ang_acc * time
        
        
    
