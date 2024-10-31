import pyglet
from pyglet import gl
import numpy as np
import trimesh
import SurfaceTension.utils as utils

import trimesh

import SurfaceTension.objects.solid as solid
import SurfaceTension.objects.fluid as fluid

# Pyglet application
window = pyglet.window.Window(800, 600, "Physics Simulation")
objects = []
time_step = 0.01

# Example: create a solid object
solid_obj = solid.SolidObject(
    name="Box",
    mass=1.0,
    volume=1.0,
    iner=np.array([1.0, 1.0, 1.0]),
    centroid=np.array([0.0, 0.0, 0.0]),
    axis=np.array([0.0, 1.0, 0.0]),
    angle=0.0,
    velo=np.array([0.0, 0.0, 0.0]),
    ang_velo=np.array([0.0, 0.0, 0.0]),
    obj_file='./data/icosahedron_input.obj'  # Replace with your .obj file path
)

objects.append(solid_obj)

@window.event
def on_draw():
    window.clear()
    for obj in objects:
        # Update the object's position and rotation
        obj.update(time_step, np.array([0, -9.81, 0]), np.array([0, 0, 0]))

        # Set up the transformation for the object
        gl.glLoadIdentity()
        gl.glTranslatef(*obj.centroid)

        # Apply the rotation matrix
        gl.glMultMatrixf(obj.rotation_matrix.T.flatten())

        # Draw the object's mesh
        for face in obj.mesh.faces:
            gl.glBegin(gl.GL_TRIANGLES)
            for vertex in face:
                gl.glVertex3f(*obj.mesh.vertices[vertex])
            gl.glEnd()

pyglet.app.run()


