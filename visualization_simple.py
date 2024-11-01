import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SurfaceTension.objects.solid as solid
import SurfaceTension.objects.fluid as fluid
import SurfaceTension.interaction_of_object.solid_solid as solid_solid
        
class SolidObjectVisualizer:
    def __init__(self, solid_objs):
        # Sorted by increasing order of mass
        self.solid_objs = sorted(solid_objs, key=lambda obj: obj.mass, reverse=False)

    def visualize(self, steps=1000, dt=0.1, forces=None, torques=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

        # Apply default forces and torques if not provided
        if forces is None:
            forces = [np.array([0, 0, 0]) for _ in self.solid_objs]
        if torques is None:
            torques = [np.array([0, 0, 0]) for _ in self.solid_objs]
        
        for _ in range(steps):
            
            ax.cla()
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])

            # energy = 0
            
            for i, solid_obj in enumerate(self.solid_objs):
                vertices, faces = solid_obj.get_mesh_data()
                # Update each object's state
                solid_obj.update(dt, forces[i] - 1 * solid_obj.centroid + np.array([0, 0, -0.981]) * solid_obj.mass, torques[i])
                
                for j in range(i - 1, -1, -1):
                    # check and handle collision
                    interaction = solid_solid.SolidSolidInteraction(solid_obj, self.solid_objs[j])
                    interaction.update()

                # Plot the centroid
                centroid = solid_obj.centroid
                ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None)

                # Draw each face
                for face in faces:
                    x = vertices[face, 0]
                    y = vertices[face, 1]
                    z = vertices[face, 2]
                    ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.7, 0.5), edgecolor='k', linewidth=0.2)

                # Orientation axes
                rotation_matrix = solid_obj.rotation_matrix
                # Draw orientation axes for each solid
                for j in range(3):
                    axis = rotation_matrix[:, j]
                    ax.quiver(centroid[0], centroid[1], centroid[2],
                              axis[0], axis[1], axis[2],
                              color=['r', 'g', 'b'][j], length=1.0)

            plt.pause(0.01)

        plt.show()

# Example: Assuming that solid_obj is your SolidObject instance and provides the getmesh_data method
solid_obj_1 = solid.SolidObject(
    name="TestSolid1",
    mass=2.0,
    volume=1.0,
    iner=np.array([1.0, 1.0, 1.0]),
    centroid=np.array([0.0, 0.0, 1.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, -0.3]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    obj_file="./data/icosahedron_input.obj"
)

solid_obj_2 = solid.SolidObject(
    name="TestSolid2",
    mass=1.0,
    volume=1.0,
    iner=np.array([1.0, 1.0, 1.0]),
    centroid=np.array([0.0, 0.0, -1.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.3]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    obj_file="./data/icosahedron_input.obj"
)

solid_objs = [solid_obj_1, solid_obj_2]

visualizer = SolidObjectVisualizer(solid_objs)
visualizer.visualize()

