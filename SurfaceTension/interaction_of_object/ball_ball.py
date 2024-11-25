import numpy as np
import trimesh
import SurfaceTension.utils as utils

class BallBallInteraction():
    def __init__(self, ball1, ball2, ela_coeff=1.0, collision_threshold=1e-3):
        self.ball1 = ball1
        self.ball2 = ball2
        # Elastic coefficient
        self.ela_coeff = ela_coeff
        self.collision_threshold = collision_threshold
        self.mass1_fraction = self.ball1.mass / (self.ball1.mass + self.ball2.mass)
        self.mass2_fraction = 1 - self.mass1_fraction


    def update(self):
        Bool_intersecting, bounce_point, normal, distance = self.check_collision()
        if (not Bool_intersecting): 
            return
        
        # print("centroid, normal: ")
        # print(self.ball1.centroid)
        # print(normal)
        
        """This part is not used anymore"""
        # # Move them to avoid the threshold, while keeping the center of mass still
        # self.ball1.centroid += self.mass2_fraction * (distance - 10 * self.collision_threshold) * normal
        # self.ball2.centroid -= self.mass1_fraction * (distance - 10 * self.collision_threshold) * normal
            
        # Vectors from centriods to bounce point 
        r_1 = normal * self.ball1.radius
        r_2 = -normal * self.ball2.radius
        
        # Velocity difference at the bounce point, v_1 - v_2
        delta_v = self.ball1.velo + np.cross(self.ball1.ang_velo, r_1) - self.ball2.velo - np.cross(self.ball2.ang_velo, r_2)
        
        # Perpendicular component
        delta_v_perp = np.dot(normal, delta_v)
        
        # if Perpendicular component is negative, this means they are going away from each other, then no collision happens (in real world)
        if delta_v_perp <= 0:
            return
        
        print("Collides!")
        print("delta_v_perp", np.dot(normal, delta_v))
        print("elastic coefficient: ", self.ela_coeff)
        
        # Inverse of Inertia tensors in ground coordinate system
        I_1_inv = self.ball1.compute_inertia_inverse_ground()
        I_2_inv = self.ball2.compute_inertia_inverse_ground()
        
        # Impluse, 1 giving to 2. (Scalar. Direction: parallel to normal vector.)
        coeff = (self.ball1.mass + self.ball2.mass) / (self.ball1.mass * self.ball2.mass) + \
                np.dot(normal, np.cross((I_1_inv @ np.cross(r_1, normal)), r_1) \
                + np.cross((I_2_inv @ np.cross(r_2, normal)), r_2))
        impulse = (((1 + self.ela_coeff) * np.dot(normal, delta_v)) / coeff) * normal
        
        # Angular impluse (according to centroids respectively)
        ang_impulse_1 = -np.cross(r_1, impulse)
        ang_impulse_2 = np.cross(r_2, impulse)
        
        # Update ball1
        self.ball1.velo -= impulse / self.ball1.mass
        self.ball1.ang_velo += I_1_inv @ ang_impulse_1
        self.ball1.is_first_frame = True
        
        # Update ball2
        self.ball2.velo += impulse / self.ball2.mass
        self.ball2.ang_velo += I_2_inv @ ang_impulse_2
        self.ball2.is_first_frame = True
        
        
    def check_collision(self):
        """
        Checks for a collision between ball1 and ball2 by calculating the minimum distance 
        between the two balls using a proximity query. If the minimum distance is less than 
        a specified threshold, the function returns True along with the collision details.
        
        Returns:
            - bool: True if a collision is detected, False otherwise.
            - bounce_point (np.array): The closest point on ball2's surface to ball1.
            - normal (np.array): The normalized direction vector from the bounce_point on ball 1 
            to the bounce_point on ball 2, representing the collision normal.
            - distance (float): The minimum distance between the two balls.
        """
        # # Initialize meshes
        # vertices1, faces1 = self.ball1.get_mesh_data()
        # vertices2, faces2 = self.ball2.get_mesh_data()

        # mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
        # mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)
        
        # # Calculate the minimum distance between ball1 and ball2
        # proximity_query = trimesh.proximity.ProximityQuery(mesh1)
        # closest_points_mesh1, closest_points_mesh2, distance = trimesh.proximity.closest_point(mesh1, mesh2)

        # # If the minimum distance is less than the collision threshold, a collision is detected
        # if np.min(distance) < self.collision_threshold:
        #     # Find the closest points on ball1 and ball2
        #     min_idx = np.argmin(distance)
            # closest_point_ball2 = closest_points_mesh2[min_idx]
            # closest_point_ball1 = closest_points_mesh1[min_idx]
            
        # Initialize meshes
        #vertices1, faces1 = self.ball1.get_mesh_data()
        #vertices2, faces2 = self.ball2.get_mesh_data()
        # print("vertices1: ", vertices1)
        # print("vertices2: ", vertices2)
        # print("faces1:", faces1)
        # print("faces2:", faces2)

        #mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
        #mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)
        # print("mesh1:", mesh1)
        
        
        """proximity_query2 = trimesh.proximity.ProximityQuery(mesh2)
        query_points1 = np.concatenate((vertices1, mesh1.sample(100)), axis=0)
        _, distances2, triangles2 = proximity_query2.on_surface(query_points1)
        # print("distances: ", distances)
        min_idx1 = np.argmin(distances2)
        distance1 = distances2[min_idx1]
        # print("distance: ", distance)
        
        proximity_query1 = trimesh.proximity.ProximityQuery(mesh1)
        query_points2 = np.concatenate((vertices2, mesh2.sample(100)), axis=0)
        _, distances1, triangles1 = proximity_query1.on_surface(query_points2)
        # print("distances: ", distances)
        min_idx2 = np.argmin(distances1)
        distance2 = distances2[min_idx2]
        # print("distance: ", distance)"""
        
        #distance = np.min([distance1, distance2])
        # Calculate the minimum distance between ball1 and ball2
        cent1 = self.ball1.centroid
        cent2 = self.ball2.centroid
        r1 = self.ball1.radius
        r2 = self.ball2.radius
        distance = np.linalg.norm(cent1 - cent2)-r1-r2
        
        #
        #closest_point_ball1 = mesh1.vertices[min_idx]
        # print("closet_point_ball1: ", closest_point_ball1)
        #closest_point_ball2 = proximity_query.on_surface([closest_point_ball1])[0][0]
        # print("closet_point_ball2: ", closest_point_ball2)
        # If the minimum distance is less than the collision threshold, a collision is detected
        if distance < self.collision_threshold:
            # Calculate the normal
            normal = cent2 - cent1
            normal = normal / np.linalg.norm(normal)
            closest_point_ball1 = cent1 + r1 * normal
            closest_point_ball2 = cent2 - r2 * normal
            
            """if distance == distance1:
                # Find the closest triangle on ball2
                triangle_id = triangles2[min_idx1]
                triangle_face = faces2[triangle_id]
                u = vertices2[triangle_face[0]]
                v = vertices2[triangle_face[1]]
                w = vertices2[triangle_face[2]]
                normal = utils.normal_of_triangle(u, v, w)
                closest_point_ball1 = query_points1[min_idx1]
                distance = np.dot(normal, closest_point_ball1 - u)
                closest_point_ball2 = closest_point_ball1 - distance * normal
                if distance > 0:
                    normal = -normal
            
            if distance == distance2:
                # Find the closest triangle on ball1
                triangle_id = triangles1[min_idx2]
                triangle_face = faces1[triangle_id]
                u = vertices1[triangle_face[0]]
                v = vertices1[triangle_face[1]]
                w = vertices1[triangle_face[2]]
                normal = utils.normal_of_triangle(u, v, w)
                closest_point_ball2 = query_points2[min_idx2]
                distance = np.dot(normal, closest_point_ball2 - u)
                closest_point_ball1 = closest_point_ball2 - distance * normal
                if distance < 0:
                    normal = -normal"""
            
            # Take weighted average
            bounce_point = self.mass1_fraction * closest_point_ball2 + self.mass2_fraction * closest_point_ball1

            # Calculate the normal as the unit vector from ball1's closest point to ball2's closest point
            #direction_vector = closest_point_ball2 - closest_point_ball1
            # print("direction_vector", direction_vector)
            # Normalize the collision normal
            #normal = direction_vector / np.linalg.norm(direction_vector)  
            # print("normal: ", normal)
            # TODO: what if normal is a zero vertor? 
            #if (distance < 0):
                #normal = -normal
            #if (distance == 0):
                # Find the normal from the face of the ball
                #normal = np.array([0, 0, 0])
            
            return True, bounce_point, normal, np.min(distance)
        return False, None, None, None