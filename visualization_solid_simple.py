import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import SurfaceTension.objects.solid as solid
#import SurfaceTension.objects.ball as ball
import SurfaceTension.objects.fluid as fluid
import SurfaceTension.interaction_of_object.solid_solid as solid_solid
import SurfaceTension.forces.force as Forces
        
class SolidObjectVisualizer:
    def __init__(self, solid_objs, ball_objs, dt, use_external_force=False, fixed_nodes=None, hanging_point=None, length=None):
        # Sorted by increasing order of mass
        self.solid_objs = sorted(solid_objs, key=lambda obj: obj.mass, reverse=False)
        self.ball_objs = sorted(ball_objs, key=lambda obj: obj.mass, reverse=False)
        self.dt = dt
        self.use_external_force = use_external_force
        if use_external_force:
            self.fixed_nodes = fixed_nodes
            self.hanging_point = hanging_point
            self.length = length
            #print("fixed_nodes:", fixed_nodes)
            #print("hanging_point:", hanging_point)
            #print("length:", length)

    def visualize(self, steps=1000, forces=None, torques=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        
        
            #forces = [Forcesget_force(obj, self.fixed_nodes[i]) for i, obj in self.ball_objs]
        # Apply default forces and torques if not provided
        if forces is None:
            forces1 = [np.array([0, 0, 0]) for _ in self.solid_objs]
            forces2 = [np.array([0, 0, 0]) for _ in self.ball_objs]
            forces = forces1 + forces2 #np.concatenate((forces1, forces2))
            #print("forces:", forces)    
        if torques is None:
            torques1 = [np.array([0, 0, 0]) for _ in self.solid_objs]
            torques2 = [np.array([0, 0, 0]) for _ in self.ball_objs]
            torques = torques1 + torques2#np.concatenate((torques1, torques2))
        
        for j in range(steps):
            
            ax.cla()
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            
            if self.use_external_force:
            #...
            
                forces_torques = [Forces.Rope("", obj, self.length, self.fixed_nodes[i], self.hanging_point, \
                 self.dt).get_force_and_torque() for i, obj in enumerate(self.ball_objs)]
                forces = [force_torque[0] for force_torque in forces_torques]
                torques = [force_torque[1] for force_torque in forces_torques]
                # print("forces:", forces)
            
            
            i = 0
            # Update solid objects
            for _, solid_obj in enumerate(self.solid_objs):
                vertices, faces = solid_obj.get_mesh_data()
                # Update each object's state
                solid_obj.update(self.dt, forces[i] + np.array([0, 0, -9.81]) * solid_obj.mass * 0, torques[i])
                #threading.Thread(target=record_data, args=(solid_obj, i)).start()
                
                with open("example_"+str(i)+".txt", 'a') as file:
                    # Write energy to file
                    #for i, solid_obj in enumerate(self.solid_objs):
                    ...#print(str(solid_obj.energy)+"\n", file=file)
                
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
                
                # Update iterator    
                i += 1
            # Update Balls
            for _, ball_obj in enumerate(self.ball_objs):
                centroid = ball_obj.centroid
                radius = ball_obj.radius    
                # Update each object's state
                #print("forces[i]:", i, forces[i])
                ball_obj.update(self.dt, forces[i] + np.array([0, 0, -9.81]) * ball_obj.mass, torques[i])
                #threading.Thread(target=record_data, args=(solid_obj, i)).start()
                
                with open("example_"+str(i)+".txt", 'a') as file:
                    # Write energy to file
                    #for i, solid_obj in enumerate(self.solid_objs):
                    ...# print(str(ball_obj.energy)+"\n", file=file)
                
                for j in range(i - 1, len(self.solid_objs) - 1, -1):
                    # check and handle collision
                    interaction = solid_solid.BallBallInteraction(ball_obj, self.ball_objs[j])
                    interaction.update()
                for j in range(len(self.solid_objs) - 1, -1, -1):
                    # check and handle collision
                    interaction = solid_solid.BallSolidInteraction(self.solid_objs[j], ball_obj)
                    interaction.update()

                # Plot the centroid
                ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None)
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 50)
                
                # Draw surface
                x = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
                y = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2] 
                ax.plot_surface(x, y, z, color='b', alpha=0.6)
                
                #Draw the hanging
                #if self.use_external_force:
                    #plt.plot(centroid+ball_obj.rotation_matrix@self.hanging_point, self.fixed_nodes[i], color='black')

                # Orientation axes
                rotation_matrix = ball_obj.rotation_matrix
                # Draw orientation axes for each solid
                for j in range(3):
                    axis = rotation_matrix[:, j]
                    ax.quiver(centroid[0], centroid[1], centroid[2],
                              axis[0], axis[1], axis[2],
                              color=['r', 'g', 'b'][j], length=1.0)
                
                # Update iterator    
                i += 1


            plt.pause(self.dt/10)

        plt.show()
        

# Example: Assuming that solid_obj is your SolidObject instance and provides the getmesh_data method
ball_obj_1 = solid.BallObject(
    name="TestSolid1",
    mass=2.0,
    volume=1.0,
    
    centroid=np.array([0.0, 0.0, 1.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, -15.0]),
    ang_velo=np.array([0.0, -2.0, 0.0]),
    radius=0.3
) #iner=np.array([1.0, 1.0, 1.0]),

ball_obj_2 = solid.BallObject(
    name="TestSolid2",
    mass=1.0,
    volume=1.0,
    
    centroid=np.array([0.0, 0.0, -1.5]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 15.0]),
    ang_velo=np.array([0.0, 2.0, 0.01]),
    radius=0.2
) #iner=np.array([25.0, 5.0, 1.0]),

ball_obj_1 = solid.BallObject(
    name="TestSolid1",
    mass=1.0,
    volume=1.0,
    
    centroid=np.array([0.0, -6.0, -0.16]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
) #iner=np.array([1.0, 1.0, 1.0]),

ball_obj_2 = solid.BallObject(
    name="TestSolid2",
    mass=1.0,
    volume=1.0,    
    centroid=np.array([0.0, 0.0, -1]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, -0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
) 

ball_obj_3 = solid.BallObject(
    name="TestSolid3",
    mass=1.0,
    volume=1.0,
    centroid=np.array([0.0, 1.0, -1]),
    axis=np.array([1.0, 0.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    radius=0.5
) 

solid_objs = []
ball_objs = [ball_obj_1, ball_obj_2, ball_obj_3]
node1 = np.array([0.0, -1.01, 9.0])
node2 = np.array([0.0, 0.0, 9.0])
node3 = np.array([0.0, 1.01, 9.0])
nodes = [node1, node2, node3]
hanging_point= np.array([0.0, 0.0, 0.0])
length = 10.0


visualizer = SolidObjectVisualizer(solid_objs, ball_objs, dt=3e-2, use_external_force=True, fixed_nodes=nodes, hanging_point=hanging_point, length=length)
visualizer.visualize()