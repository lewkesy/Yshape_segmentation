import os
import sys

from YTree.curve import get_curve, random_sample_circle, sample_circle
import numpy as np
from IPython import embed
from YTree.utils.visualization import save_Yshape_ply, save_mesh

def vector_normalize(v, axis=1):
    return v / np.sqrt(np.sum(v ** 2, axis=axis, keepdims=True))


def rodrigues_rotation(axis_vec, dir, theta):
    return dir*np.cos(theta) + np.cross(axis_vec, dir, axis=1)*np.sin(theta) + axis_vec*np.sum(axis_vec*dir, axis=1)*(1-np.cos(theta))


def gen_dir(d_root, root, dst):
    d = dst - root
    axis_vec = vector_normalize(np.cross(d, d_root), axis=1)
    theta = (np.random.rand(1)[0]*60-30) / 180 * np.pi
    
    direction = rodrigues_rotation(axis_vec, d_root, theta)

    return direction


def vector_transform(vec, rotation, transformation):
    if len(vec.shape) == 2:
        vec = vec[:, :, None]
    transformed_vec =  np.matmul(rotation, vec)[:, :, 0] + transformation[:, 0, :]

    return transformed_vec


def visualize_yshape_point_cloud(pos, dir, radius, name):

    root, left, right = pos
    d_root, d_left, d_right = vector_normalize(dir)
    r_root, r_left, r_right = radius

    left_points, left_dirs, left_radius = get_curve(root, left, d_root, d_left, r_root, r_left)
    right_points, right_dirs, right_radius = get_curve(root, right, d_root, d_right, r_root, r_right)

    left_sampled_points = random_sample_circle(left_points[1:], left_dirs[1:], left_radius[1:])
    right_sampled_points = random_sample_circle(right_points[1:], right_dirs[1:], right_radius[1:])

    points = np.concatenate((left_sampled_points, right_sampled_points), axis=0)

    save_Yshape_ply('visualize/yshape/%s.ply' % name, points)


def gen_mesh(points, dirs, rs, start_idx):

    sample_num = 100
    total_points = []
    total_faces = []
    
    # sample all nodes for each circle
    for point, dir, r in zip(points, dirs, rs):
        
        sampled_points = sample_circle(point, dir, r, sample_num)
        total_points.append(sampled_points)
    
    # find two layers for mesh generation
    for layer in range(len(total_points)-1):

        node = layer * sample_num + start_idx
        for i in range(sample_num-1):

            # find the start node for the mesh
            total_faces.append([3, node, node + sample_num + 1, node + sample_num])
            total_faces.append([3, node, node + 1, node + sample_num + 1])
            node += 1
        
        # find the last mesh for this layer
        total_faces.append([3, node, node + sample_num + 1 - sample_num, node + sample_num])
        total_faces.append([3, node, node + 1 - sample_num, node + sample_num + 1 - sample_num])
    

    return np.array(total_points), np.array(total_faces)


def visualize_yshape_mesh(pos, dir, radius, name):
    root, left, right = pos
    dir[:, 0] += 1e-6
    d_root, d_left, d_right = vector_normalize(dir)
    r_root, r_left, r_right = radius

    left_points, left_dirs, left_radius = get_curve(root, left, d_root, d_left, r_root, r_left, scalar=20)
    right_points, right_dirs, right_radius = get_curve(root, right, d_root, d_right, r_root, r_right, scalar=20)

    mesh_points = []
    mesh_faces = []

    start_idx = 0
    left_mesh_points, left_mesh_faces = gen_mesh(left_points[1:], left_dirs[1:], left_radius[1:], start_idx)
    mesh_points.append(left_mesh_points)
    mesh_faces.append(left_mesh_faces)

    start_idx += left_mesh_points.shape[0] * left_mesh_points.shape[1]
    right_mesh_points, right_mesh_faces = gen_mesh(right_points[1:], right_dirs[1:], right_radius[1:], start_idx)
    mesh_points.append(right_mesh_points)
    mesh_faces.append(right_mesh_faces)

    mesh_points = np.array(mesh_points).reshape(-1, 3)
    mesh_faces = np.array(mesh_faces).reshape(-1, 4)

    save_mesh(name, mesh_points, mesh_faces)


if __name__ == "__main__":

    pos = np.array([[0, -1, 0] , [1, 0, 0], [-1, 0, 0]]).astype(np.float)
    dir = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0]]).astype(np.float)
    r = np.array([0.2, 0.2, 0.2])
    visualize_yshape_mesh(pos, dir, r, "test.ply")