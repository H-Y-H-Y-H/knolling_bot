from stl import mesh
import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def calculate_mesh_centroid(stl_mesh):
    # Calculate the centroid of the entire mesh
    return np.mean(stl_mesh.points.reshape(-1, 3), axis=0)

def apply_rotation(stl_mesh, axis, angle_degree):
    angle_rad = np.deg2rad(angle_degree)
    rot_matrix = rotation_matrix(axis, angle_rad)
    
    centroid = calculate_mesh_centroid(stl_mesh)
    stl_mesh.vectors = np.dot(stl_mesh.vectors - centroid, rot_matrix) + centroid
    return stl_mesh

# Specify the STL file path and the rotation parameters
stl_file = 'electronics_stl/utilityknife/utilityknife_1.stl'
output_file = 'electronics_stl/utilityknife/utilityknife_rotated.stl'
rotation_axis = [0, 1, 0] # x, y, z axis
rotation_angle = 270

# Load the STL file
stl_mesh = mesh.Mesh.from_file(stl_file)

# Apply the rotation
stl_mesh = apply_rotation(stl_mesh, rotation_axis, rotation_angle)

# Save the rotated STL
stl_mesh.save(output_file)

