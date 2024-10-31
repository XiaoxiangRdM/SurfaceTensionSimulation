import numpy as np
import trimesh

class SolidSolidInteraction():
    def __init__(self, solid1, solid2, ela_coeff=1.0, collision_threshold=1e-3):
        self.solid1 = solid1
        self.solid2 = solid2
        # Elastic coefficient
        self.ela_coeff = ela_coeff
        self.collision_threshold = collision_threshold
        self.mass1_fraction = self.solid1.mass / (self.solid1.mass + self.solid2.mass)
        self.mass2_fraction = 1 - self.mass1_fraction


    def update(self, time):
        Bool_intersecting, bounce_point, normal, distance = self.check_collision()
        if (not Bool_intersecting): 
            return
        
        if (distance < 0): 
            # Move them to avoid the threshold, while keeping the center of mass still
            self.solid1.centroid += self.mass2_fraction * (distance + 2 * self.collision_threshold) * normal
            self.solid2.centroid -= self.mass1_fraction * (distance + 2 * self.collision_threshold) * normal
            
        # Vectors from centriods to bounce point 
        r_1 = bounce_point - self.solid1.centroid
        r_2 = bounce_point - self.solid2.centroid
        
        # Velocity difference at the bounce point, v_1 - v_2
        delta_v = self.solid1.velo + np.cross(self.solid1.ang_velo, r_1) - self.solid2.velo - np.cross(self.solid2.ang_velo, r_2)
        
        # Inverse of Inertia tensors in ground coordinate system
        I_1_inv = self.solid1.compute_inertia_inverse_ground()
        I_2_inv = self.solid2.compute_inertia_inverse_ground()
        
        # Impluse, 1 giving to 2. (Scalar. Direction: parallel to normal vector.)
        coeff = (self.solid1.mass + self.solid2.mass) / (self.solid1.mass * self.solid2.mass) + \
                np.dot(normal, np.cross((I_1_inv @ np.cross(r_1, normal)), r_1) \
                + np.cross((I_2_inv @ np.cross(r_2, normal)), r_2))
        impulse = (((1 + self.ela_coeff) * np.dot(normal, delta_v)) / coeff) * normal
        
        # Angular impluse (according to centroids respectively)
        ang_impulse_1 = -np.cross(r_1, impulse)
        ang_impulse_2 = np.cross(r_2, impulse)
        
        # Update solid1
        self.solid1.velo += impulse / self.solid1.mass
        self.solid1.ang_velo += I_1_inv @ ang_impulse_1
        
        # Update solid2
        self.solid2.velo += impulse / self.solid2.mass
        self.solid2.ang_velo += I_2_inv @ ang_impulse_2
        
        
    def check_collision(self):
        """
        Checks for a collision between solid1 and solid2 by calculating the minimum distance 
        between the two solids using a proximity query. If the minimum distance is less than 
        a specified threshold, the function returns True along with the collision details.
        
        Returns:
            - bool: True if a collision is detected, False otherwise.
            - bounce_point (np.array): The closest point on solid2's surface to solid1.
            - normal (np.array): The normalized direction vector from the bounce_point on solid1 
            to the bounce_point on solid 2, representing the collision normal.
            - distance (float): The minimum distance between the two solids.
        """
        # Calculate the minimum distance between solid1 and solid2
        proximity_query = trimesh.proximity.ProximityQuery(self.solid1)
        closest_points_solid2, distance = proximity_query.signed_distance(self.solid2)

        # If the minimum distance is less than the collision threshold, a collision is detected
        if np.min(distance) < self.collision_threshold:
            # Find the closest points on solid1 and solid2
            min_idx = np.argmin(distance)
            closest_point_solid2 = closest_points_solid2[min_idx]
            closest_point_solid1 = proximity_query.closest_point(self.solid1, closest_point_solid2)
            
            # Take weighted average
            bounce_point = self.mass1_fraction * closest_point_solid2 + self.mass2_fraction * closest_point_solid1

            # Calculate the normal as the unit vector from solid1's closest point to solid2's closest point
            direction_vector = closest_point_solid2 - closest_point_solid1
            # Normalize the collision normal
            normal = direction_vector / np.linalg.norm(direction_vector)  
            # TODO: what if normal is a zero vertor? 
            
            return True, bounce_point, normal, np.min(distance)
        return False, None, None, None
