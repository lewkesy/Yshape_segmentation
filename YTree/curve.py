import numpy as np
from IPython import embed

def get_curve(ps, pe, ds, de, rs, re, scalar=100, axis=-1):
    
    ps_control = ps + ds
    pe_control = pe - de

    samples = [i/scalar for i in range(scalar)]

    points = []
    dirs = []
    radius = []
    for t in samples:
        p = (1-t)**3 * ps + 3 * (1-t)**2*t * ps_control + 3 * (1-t)*t**2 * pe_control + t**3 * pe
        d = 3 * (1-t)**2 * (ps_control - ps) + 6 * (1-t)*t * (pe_control - ps_control) + 3 * t**2 * (pe - pe_control)
        d = d / np.sqrt(np.sum(d**2, keepdims=True, axis=axis))
        r = rs * (1-t) + re * t

        points.append(p)
        dirs.append(d)
        radius.append(r)

    points = np.array(points)
    dirs = np.array(dirs)
    radius = np.array(radius)

    return points, dirs, radius


def sample_circle(point, dir, r, sample_num=20):

    '''
    input:
    points: 3
    dirs  : 3
    r     : 1

    '''

    x_bar = np.cross(point, dir)
    x_bar /= np.sqrt(np.sum(x_bar**2))

    y_bar = np.cross(dir, x_bar)
    y_bar /= np.sqrt(np.sum(y_bar**2))

    theta = np.array([i/sample_num for i in range(sample_num)]) * 2 * np.pi

    sampled_points = point[None, :] + r * (np.cos(theta)[:, None] * x_bar[None, :] + np.sin(theta)[:, None] * y_bar[None, :])
    sampled_points = sampled_points.reshape(-1, 3)

    return sampled_points


def random_sample_circle(point, dir, r, sample_num=20, axis=-1):

    '''
    input:
    points: s, 3
    dirs  : s, 3
    r     : s

    '''
    skeleton_sample, _ = point.shape
    point = point.reshape(-1, 3)
    dir = dir.reshape(-1, 3)
    r = r.reshape(-1, 1)

    x_bar = np.cross(point, dir)
    x_bar /= np.sqrt(np.sum(x_bar**2, axis=axis, keepdims=True))

    y_bar = np.cross(dir, x_bar)
    y_bar /= np.sqrt(np.sum(y_bar**2, axis=axis, keepdims=True))

    random_r = np.sqrt(np.random.rand(skeleton_sample, sample_num)) * r
    random_theta = np.random.rand(skeleton_sample, sample_num) * 2 * np.pi

    sampled_points = point[:, None, :] + random_r[:, :, None] * (np.cos(random_theta)[:, :, None] * x_bar[:, None, :] + np.sin(random_theta)[:, :, None] * y_bar[:, None, :])
    sampled_points = sampled_points.reshape(-1, 3)

    return sampled_points
