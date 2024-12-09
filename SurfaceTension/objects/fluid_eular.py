import numpy as np
import trimesh
import SurfaceTension.utils as utils

class Fluid_product_eular():
    def __init__(self, name, posi, dens, x_1, x_2, dx, dt, height, velo, g=9.8, beta=1):
        """
        Initialize the FluidObject using Eular method.

        Parameters:
        - name (str): Name of the object.
        - posi (np.ndarray): Position where the fluid starts.
        - dens (float): Density of the object.
        - x_1 (int): Length of the fluis in the first axis, use number of grids to represent.
        - x_2 (int): Length of the fluis in the second axis, use number of grids to represent.
        - dx (float): The length of the grid.
        - dt (float): The time step.
        - height (np.ndarray): The height field of the fluid.
        - height_last (np.ndarray): The height field of the fluid in the last frame.
        - velo (np.ndarray): The velocity field of the fluid.
        - g (float): The gravity acceleration.
        - beta (float): The coefficient of viscosity.
        """
        assert dens > 0, "Density must be a float larger than zero."
        assert height.shape == (x_1, x_2), "Height field must be a 2D array."
        assert velo.shape == (x_1, x_2, 2), "Velocity field must be a 3D array."
        self.name = name
        self.posi = posi
        self.dens = dens
        self.x_1 = x_1
        self.x_2 = x_2
        self.dx = dx
        self.dt = dt
        self.height = height
        self.height_last = height
        self.velo = velo
        self.g = g
        self.beta = beta
        
    def update(self):
        """
        Update the fluid object using Eular method.
        """
        new_height = np.zeros((self.x_1, self.x_2))
        for i in range(1, self.x_1-1):
            for j in range(1, self.x_2-1):
                new_height[i][j] = self.height[i][j] + self.beta * (self.height[i][j] - self.height_last[i][j])
                new_height[i][j] += self.dt**2 * self.height[i][j] * self.g / (self.dx**2) \
                    * (self.height[i+1][j] + self.height[i-1][j] + self.height[i][j+1] + self.height[i][j-1] - 4 * self.height[i][j])
        # Boundary conditions
        for j in range(1, self.x_2-1):
            new_height[0][j] = self.height[0][j] + self.beta * (self.height[0][j] - self.height_last[0][j]) 
            new_height[0][j] += self.dt**2 * self.height[0][j] * self.g / (self.dx**2) \
                * (self.height[1][j] + self.height[0][j+1] + self.height[0][j-1] - 3 * self.height[0][j])
            new_height[self.x_1-1][j] = self.height[self.x_1-1][j] + self.beta * (self.height[self.x_1-1][j] - self.height_last[self.x_1-1][j])  
            new_height[self.x_1-1][j] += self.dt**2 * self.height[self.x_1-1][j] * self.g / (self.dx**2) \
                * (self.height[self.x_1-2][j] + self.height[self.x_1-1][j+1] + self.height[self.x_1-1][j-1] - 3 * self.height[self.x_1-1][j])
        for i in range(1, self.x_1-1):
            new_height[i][0] = self.height[i][0] + self.beta * (self.height[i][0] - self.height_last[i][0])   
            new_height[i][0] += self.dt**2 * self.height[i][0] * self.g / (self.dx**2) \
                * (self.height[i][1] + self.height[i+1][0] + self.height[i-1][0] - 3 * self.height[i][0])
            new_height[i][self.x_2-1] = self.height[i][self.x_2-1] + self.beta * (self.height[i][self.x_2-1] - self.height_last[i][self.x_2-1])   
            new_height[i][self.x_2-1] += self.dt**2 * self.height[i][self.x_2-1] * self.g / (self.dx**2) \
                * (self.height[i][self.x_2-2] + self.height[i+1][self.x_2-1] + self.height[i-1][self.x_2-1] - 3 * self.height[i][self.x_2-1])
        # Corner conditions
        new_height[0][0] = self.height[0][0] + self.beta * (self.height[0][0] - self.height_last[0][0])
        new_height[0][0] += self.dt**2 * self.height[0][0] * self.g / (self.dx**2) \
            * (self.height[1][0] + self.height[0][1] - 2 * self.height[0][0])
        new_height[self.x_1-1][0] = self.height[self.x_1-1][0] + self.beta * (self.height[self.x_1-1][0] - self.height_last[self.x_1-1][0])    
        new_height[self.x_1-1][0] += self.dt**2 * self.height[self.x_1-1][0] * self.g / (self.dx**2) \
            * (self.height[self.x_1-2][0] + self.height[self.x_1-1][1] - 2 * self.height[self.x_1-1][0])
        new_height[0][self.x_2-1] = self.height[0][self.x_2-1] + self.beta * (self.height[0][self.x_2-1] - self.height_last[0][self.x_2-1])
        new_height[0][self.x_2-1] += self.dt**2 * self.height[0][self.x_2-1] * self.g / (self.dx**2) \
            * (self.height[1][self.x_2-1] + self.height[0][self.x_2-2] - 2 * self.height[0][self.x_2-1])
        new_height[self.x_1-1][self.x_2-1] = self.height[self.x_1-1][self.x_2-1] + self.beta * (self.height[self.x_1-1][self.x_2-1] - self.height_last[self.x_1-1][self.x_2-1])
        new_height[self.x_1-1][self.x_2-1] += self.dt**2 * self.height[self.x_1-1][self.x_2-1] * self.g / (self.dx**2) \
            * (self.height[self.x_1-2][self.x_2-1] + self.height[self.x_1-1][self.x_2-2] - 2 * self.height[self.x_1-1][self.x_2-1])
        # Update the stored data
        self.height_last = self.height
        self.height = new_height
            