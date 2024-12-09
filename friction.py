import numpy as np

class Friction():
    def check_interaction(self, obj1, obj2):
        # Check if the object 1 and 2 are in contact with each other
        # If it is, return True
        # If it is not, return False
        self.obj1 = obj1
        self.obj2 = obj2
        pass 
        return True
    
    def calculate_friction(self):
        # Calculate the friction between the object 1 and 2
        # Return the friction value
        
        alpha = (self.obj1.mass + self.obj2.mass)/(self.obj1.mass * self.obj2.mass) 
        M = alpha * np.identity(3) - ([r]*self.obj1.iner_inv*[r] + [r]*self.obj2.iner_inv*[r])
        p = (1+e) 