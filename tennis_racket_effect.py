import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import SurfaceTension.objects.solid as solid
import SurfaceTension.objects.fluid as fluid
import SurfaceTension.interaction_of_object.solid_solid as solid_solid
        
class SolidObjectVisualizer:
    def __init__(self, solid_objs, dt):
        # Sorted by increasing order of mass
        self.solid_objs = sorted(solid_objs, key=lambda obj: obj.mass, reverse=False)
        self.dt = dt

    def visualize(self, steps=1000):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        ani = animation.FuncAnimation(fig, self.update, frames=steps, interval=self.dt, blit=True, fargs=(ax,))
        ani.save('tennis_racket.mp4', writer='ffmpeg', fps=30)
        
        #for _ in range(steps):
    def update(self, frame, ax):
            
            ax.cla()
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            
            artists = []
            
            for i, solid_obj in enumerate(self.solid_objs):
                vertices, faces = solid_obj.get_mesh_data()
                # Update each object's state
                solid_obj.update(self.dt, 0 , 0)
                # print("Energy: ", solid_obj.energy)
                
                for j in range(i - 1, -1, -1):
                    # check and handle collision
                    interaction = solid_solid.SolidSolidInteraction(solid_obj, self.solid_objs[j])
                    interaction.update()

                # Plot the centroid
                centroid = solid_obj.centroid
                artists.append(ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None))

                # Draw each face
                for face in faces:
                    x = vertices[face, 0]
                    y = vertices[face, 1]
                    z = vertices[face, 2]
                    artists.append(ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.7, 0.5), edgecolor='k', linewidth=0.2))

                # Orientation axes
                rotation_matrix = solid_obj.rotation_matrix
                # Draw orientation axes for each solid
                for j in range(3):
                    axis = rotation_matrix[:, j]
                    artists.append(ax.quiver(centroid[0], centroid[1], centroid[2],
                              axis[0], axis[1], axis[2],
                              color=['r', 'g', 'b'][j], length=1.0))

            #plt.pause(self.dt)
            return artists

        #plt.show()

# Example: Assuming that solid_obj is your SolidObject instance and provides the getmesh_data method
solid_obj_1 = solid.SolidObject(
    name="TestSolid1",
    mass=2.0,
    volume=1.0,
    iner=np.array([25.0, 5.0, 1.0]),
    centroid=np.array([0.0, 0.0, 0.0]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 1.0, 0.01]),
    obj_file="./data/icosahedron_input.obj"
)

solid_objs = [solid_obj_1]

visualizer = SolidObjectVisualizer(solid_objs, dt=3e-2)
visualizer.visualize()

