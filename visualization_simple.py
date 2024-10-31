import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SurfaceTension.objects.solid as solid
import SurfaceTension.objects.fluid as fluid

class SolidObjectVisualizer:
    def __init__(self, solid_obj):
        self.solid_obj = solid_obj
        self.vertices, self.faces = self.solid_obj.get_mesh_data()  # Read grid data

    def visualize(self, steps=100, dt=0.1, force=np.array([0, 0, 0]), torque=np.array([0.0, 0.0, 0.0])):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

        for _ in range(steps):
            # Update the object state
            self.solid_obj.update(dt, -1 * self.solid_obj.centroid, torque)

            # Clear previous plot
            ax.cla()
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_zlim([-5, 5])

            # Plot the centroid
            centroid = self.solid_obj.centroid
            ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid")

            # Draw orientation axes
            rotation_matrix = self.solid_obj.rotation_matrix
            rotated_vertices = (rotation_matrix @ self.vertices.T).T + centroid  # Rotate and translate vertices

            # Draw each face
            for face in self.faces:
                x = rotated_vertices[face, 0]
                y = rotated_vertices[face, 1]
                z = rotated_vertices[face, 2]
                ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.7, 0.5), edgecolor='k', linewidth=0.2)

            # Draw axes
            for i in range(3):
                axis = rotation_matrix[:, i]
                ax.quiver(centroid[0], centroid[1], centroid[2], 
                          axis[0], axis[1], axis[2], 
                          color=['r', 'g', 'b'][i], length=1.0)

            plt.pause(0.01)

        plt.show()

# Example: Assuming that solid_obj is your SolidObject instance and provides the getmesh_data method
solid_obj = solid.SolidObject(
    name="TestSolid",
    mass=1.0,
    volume=1.0,
    iner=np.array([1.0, 1.0, 10.0]),
    centroid=np.array([0.0, 0.0, 1.0]),
    axis=np.array([0.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 1.0, 0.0]),
    ang_velo=np.array([0.1, 0.0, 0.1]),
    obj_file="./data/icosahedron_input.obj"
)

visualizer = SolidObjectVisualizer(solid_obj)
visualizer.visualize()

