import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import threading
import SurfaceTension.objects.solid as solid
import SurfaceTension.objects.fluid as fluid
import SurfaceTension.interaction_of_object.solid_solid as solid_solid
import SurfaceTension.forces.force as Forces


class SolidObjectVisualizer:
    def __init__(self, solid_objs, ball_objs, dt, use_external_force=False, fixed_nodes=None, hanging_point=None, length=None):
        # Sort the solid and ball objects by increasing order of mass
        self.solid_objs = sorted(solid_objs, key=lambda obj: obj.mass, reverse=False)
        self.ball_objs = sorted(ball_objs, key=lambda obj: obj.mass, reverse=False)

        self.dt = dt
        self.use_external_force = use_external_force

        if use_external_force:
            self.fixed_nodes = fixed_nodes
            self.hanging_point = hanging_point
            self.length = length

    def update(self, frame, ax, forces, torques):
        ax.cla()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

        artists = []

        if self.use_external_force:
            forces_torques = [
                Forces.Rope("", obj, self.length, self.fixed_nodes[i], self.hanging_point, self.dt).get_force_and_torque()
                for i, obj in enumerate(self.ball_objs)
            ]
            forces = [force_torque[0] for force_torque in forces_torques]
            torques = [force_torque[1] for force_torque in forces_torques]

        i = 0

        # Update solid objects
        for solid_obj in self.solid_objs:
            vertices, faces = solid_obj.get_mesh_data()

            # Update each solid object's state
            solid_obj.update(self.dt, forces[i] + np.array([0, 0, -9.81]) * solid_obj.mass * 0, torques[i])

            # Log energy to file
            with open(f"example_{i}.txt", 'a') as file:
                # Print energy data for solid object to file if needed
                # print(str(solid_obj.energy) + "\n", file=file)
                pass

            # Check for collisions with other solids
            for j in range(i - 1, -1, -1):
                interaction = solid_solid.SolidSolidInteraction(solid_obj, self.solid_objs[j])
                interaction.update()

            # Plot centroid
            centroid = solid_obj.centroid
            artists.append(ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None))

            # Draw faces
            for face in faces:
                x = vertices[face, 0]
                y = vertices[face, 1]
                z = vertices[face, 2]
                artists.append(ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.7, 0.5), edgecolor='k', linewidth=0.2))

            # Draw orientation axes
            rotation_matrix = solid_obj.rotation_matrix
            for j in range(3):
                axis = rotation_matrix[:, j]
                artists.append(ax.quiver(centroid[0], centroid[1], centroid[2], axis[0], axis[1], axis[2], color=['r', 'g', 'b'][j], length=1.0))

            i += 1

        # Update ball objects
        for ball_obj in self.ball_objs:
            centroid = ball_obj.centroid
            radius = ball_obj.radius

            # Update ball object's state
            ball_obj.update(self.dt, forces[i] + np.array([0, 0, -9.81]) * ball_obj.mass, torques[i])

            # Log energy to file
            with open(f"example_{i}.txt", 'a') as file:
                # Print energy data for ball object to file if needed
                # print(str(ball_obj.energy) + "\n", file=file)
                pass

            # Check for collisions with other balls
            for j in range(i - 1, len(self.solid_objs) - 1, -1):
                interaction = solid_solid.BallBallInteraction(ball_obj, self.ball_objs[j])
                interaction.update()

            # Check for collisions with solids
            for j in range(len(self.solid_objs) - 1, -1, -1):
                interaction = solid_solid.BallSolidInteraction(self.solid_objs[j], ball_obj)
                interaction.update()

            # Plot centroid
            artists.append(ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None))

            # Draw surface of ball
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]
            artists.append(ax.plot_surface(x, y, z, color='b', alpha=0.6))
            
            #Plot line from fixed node to hanging point
            rotation_matrix = ball_obj.rotation_matrix
            artists.append(ax.plot([self.fixed_nodes[i][0], centroid[0] + (rotation_matrix@self.hanging_point)[0]], \
                                   [self.fixed_nodes[i][1], centroid[1] + (rotation_matrix@self.hanging_point)[1]], \
                                   [self.fixed_nodes[i][2], centroid[2] + (rotation_matrix@self.hanging_point)[2]], color='r', linewidth=2))

            # Draw orientation axes
            for j in range(3):
                axis = rotation_matrix[:, j]
                artists.append(ax.quiver(centroid[0], centroid[1], centroid[2], axis[0], axis[1], axis[2], color=['r', 'g', 'b'][j], length=1.0))

            i += 1

        return artists

    def visualize(self, steps=1000, forces=None, torques=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])

        if forces is None:
            forces = [np.array([0, 0, 0]) for _ in range(len(self.solid_objs) + len(self.ball_objs))]

        if torques is None:
            torques = [np.array([0, 0, 0]) for _ in range(len(self.solid_objs) + len(self.ball_objs))]

        # Create animation
        ani = animation.FuncAnimation(fig, self.update, frames=steps, interval=self.dt / 10, blit=True, fargs=(ax, forces, torques))

        # Save the animation to a video file
        ani.save('newton_withline.mp4', writer='ffmpeg', fps=30)

        plt.show()


# Example: Create ball objects and solid objects, then visualize
ball_obj_1 = solid.BallObject(
    name="TestBall1",
    mass=1.0,
    volume=1.0,
    centroid=np.array([0.0, -6.0, -0.16]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
)

ball_obj_2 = solid.BallObject(
    name="TestBall2",
    mass=1.0,
    volume=1.0,
    centroid=np.array([0.0, 0.0, -2.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
)

ball_obj_3 = solid.BallObject(
    name="TestBall3",
    mass=1.0,
    volume=1.0,
    centroid=np.array([0.0, 1.1, -1.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
)

ball_objs = [ball_obj_1, ball_obj_2, ball_obj_3]
solid_objs = []  # Add solid objects if needed

nodes = [np.array([0.0, -1.01, 9.0]), np.array([0.0, 0.0, 9.0]), np.array([0.0, 1.01, 9.0])]
hanging_point = np.array([0.0, 0.0, 0.0])
length = 10.0

visualizer = SolidObjectVisualizer(solid_objs, ball_objs, dt=3e-2, use_external_force=True, fixed_nodes=nodes, hanging_point=hanging_point, length=length)
visualizer.visualize()
