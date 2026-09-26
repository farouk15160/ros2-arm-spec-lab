"""Calibrated RGB/depth perception with explicit external algorithm boundaries."""
from importlib import import_module
import numpy as np

from .config_contract import fields, number, text, version
from .scene_config import mutable


def depth_to_pointcloud(depth, intrinsics, *, depth_scale=1., min_depth=.01, max_depth=10.):
    """Unproject registered depth into optical frame (+x right, +y down, +z forward)."""
    image = np.asarray(depth, dtype=float)
    if image.ndim != 2 or not image.size:
        raise ValueError('depth: expected nonempty HxW image')
    for key in ('fx', 'fy', 'cx', 'cy'):
        number(intrinsics.get(key), 'intrinsics.' + key, positive=key in ('fx', 'fy'))
    number(depth_scale, 'depth_scale', positive=True)
    if not (np.isfinite([min_depth, max_depth]).all() and 0 <= min_depth < max_depth):
        raise ValueError('depth limits: expected 0 <= min_depth < max_depth')
    z = image * depth_scale
    rows, cols = np.indices(image.shape)
    valid = np.isfinite(z) & (z > 0) & (z >= min_depth) & (z <= max_depth)
    return np.column_stack(((cols[valid] - intrinsics['cx']) * z[valid] / intrinsics['fx'],
                            (rows[valid] - intrinsics['cy']) * z[valid] / intrinsics['fy'], z[valid]))


def filter_pointcloud(points, *, bounds=None, voxel_size=None, transform=None):
    """Transform points, crop in destination frame and retain one centroid per voxel."""
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError('points: expected Nx3 array')
    cloud = cloud[np.isfinite(cloud).all(axis=1)].copy()
    if transform is not None:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError('transform: expected finite homogeneous 4x4 matrix')
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError('transform: rotation must be orthonormal with determinant +1')
        cloud = cloud @ rotation.T + matrix[:3, 3]
    if bounds is not None:
        box = np.asarray(bounds, dtype=float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
            raise ValueError('bounds: expected finite lower and upper XYZ')
        cloud = cloud[np.all((cloud >= box[0]) & (cloud <= box[1]), axis=1)]
    if voxel_size is not None:
        number(voxel_size, 'voxel_size', positive=True)
        if len(cloud):
            _, inverse = np.unique(np.floor(cloud / voxel_size), axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            cloud = np.column_stack([np.bincount(inverse, weights=cloud[:, axis]) / counts
                                     for axis in range(3)])
    return cloud


def validate_perception(data):
    modules = ('pointcloud', 'octomap', 'slam', 'object_recognition', 'image_policy')
    fields(data, ('schema_version', 'enabled'), modules, path='perception')
    version(data['schema_version'], 'perception')
    if type(data['enabled']) is not bool:
        raise ValueError('perception.enabled: expected boolean')
    for name in modules:
        if name not in data:
            continue
        spec = data[name]
        options = {'pointcloud': ('camera', 'min_depth', 'max_depth', 'voxel_size', 'target_frame'),
                   'octomap': ('resolution', 'max_range')}.get(name, ('plugin', 'parameters', 'camera'))
        fields(spec, ('enabled',), options, path='perception.' + name)
        if type(spec['enabled']) is not bool:
            raise ValueError(name + '.enabled: expected boolean')
        if not spec['enabled']:
            continue
        if name == 'pointcloud':
            fields(spec, ('enabled',) + options, path='perception.pointcloud')
            for key in ('camera', 'target_frame'):
                text(spec[key], 'pointcloud.' + key)
            for key in ('min_depth', 'max_depth', 'voxel_size'):
                number(spec[key], 'pointcloud.' + key, positive=True)
            if spec['min_depth'] >= spec['max_depth']:
                raise ValueError('pointcloud: min_depth must be below max_depth')
        elif name == 'octomap':
            fields(spec, ('enabled',) + options, path='perception.octomap')
            for key in options:
                number(spec[key], 'octomap.' + key, positive=True)
            if not data.get('pointcloud', {}).get('enabled'):
                raise ValueError('octomap: requires enabled pointcloud')
        else:
            text(spec.get('plugin'), name + '.plugin')
            text(spec.get('camera'), name + '.camera')
            if ':' not in spec['plugin']:
                raise ValueError(name + '.plugin: expected module:callable')
            if not isinstance(spec.get('parameters', {}), dict):
                raise ValueError(name + '.parameters: expected mapping')


def validate_perception_dependencies(project):
    from .scene_config import resolve_sensors
    data = mutable(project.sections.get('perception', {'schema_version': 1, 'enabled': False}))
    validate_perception(data)
    if not data['enabled']:
        return
    cameras = {item['name']: item for item in resolve_sensors(project)}
    for name, spec in data.items():
        if isinstance(spec, dict) and spec.get('enabled') and name != 'octomap':
            camera = cameras.get(spec['camera'])
            if camera is None:
                raise ValueError(f'perception.{name}: enabled camera {spec["camera"]} is required')
            if name == 'pointcloud' and camera['type'] == 'rgb':
                raise ValueError('perception.pointcloud: depth or rgbd camera is required')
            if name in ('object_recognition', 'image_policy') and camera['type'] == 'depth':
                raise ValueError('perception.' + name + ': RGB camera is required')


def moveit_octomap_config(data):
    data = mutable(data)
    validate_perception(data)
    if not data['enabled'] or not data.get('octomap', {}).get('enabled'):
        return {'sensors': []}
    spec = data['octomap']
    return {'octomap_resolution': float(spec['resolution']), 'octomap_frame': data['pointcloud']['target_frame'],
            'sensors': ['depth_cloud'], 'depth_cloud': {
                'sensor_plugin': 'occupancy_map_monitor/PointCloudOctomapUpdater',
                'point_cloud_topic': '/perception/octomap_points', 'max_range': float(spec['max_range']),
                'point_subsample': 1, 'padding_offset': .01, 'padding_scale': 1., 'max_update_rate': 5.0,
                'filtered_cloud_topic': '/perception/filtered_points'}}


def load_algorithm(reference):
    """Explicitly installed plugin callable(frame_bundle, parameters) -> result.

    Plugins execute trusted local Python. Configuration must be trusted as code.
    A frame bundle has stamp, frame_id, rgb/depth and intrinsics where available.
    """
    if reference == 'builtin:color_components':
        return lambda frame, params: color_components(frame['rgb'], params)
    try:
        module, attribute = reference.split(':', 1)
        algorithm = getattr(import_module(module), attribute)
    except (ValueError, ImportError, AttributeError) as exc:
        raise RuntimeError(f'perception plugin {reference!r} unavailable: {exc}') from exc
    if not callable(algorithm):
        raise RuntimeError(f'perception plugin {reference!r} must be callable')
    return algorithm


def color_components(rgb, parameters):
    """Baseline connected color regions, not trained semantic object recognition."""
    image = np.asarray(rgb)
    lower, upper = (np.asarray(parameters.get(k), dtype=float) for k in ('lower_rgb', 'upper_rgb'))
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('rgb: expected uint8 HxWx3 image')
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(lower < 0) or np.any(upper > 255) or np.any(lower > upper):
        raise ValueError('color thresholds: expected ordered RGB bounds in [0,255]')
    minimum = parameters.get('min_pixels', 1)
    if type(minimum) is not int or minimum < 1:
        raise ValueError('min_pixels: expected positive integer')
    mask = np.all((image >= lower) & (image <= upper), axis=2)
    seen, result = set(), []
    for row, col in zip(*np.nonzero(mask)):
        if (row, col) in seen:
            continue
        pending, pixels = [(row, col)], []
        while pending:
            y, x = pending.pop()
            if (y, x) in seen:
                continue
            seen.add((y, x))
            pixels.append((y, x))
            pending.extend((ny, nx) for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                           if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                           and mask[ny, nx] and (ny, nx) not in seen)
        if len(pixels) >= minimum:
            points = np.asarray(pixels)
            low, high = points.min(axis=0), points.max(axis=0)
            result.append({'label': 'color_region', 'pixels': len(pixels),
                           'bbox_xyxy': [int(low[1]), int(low[0]), int(high[1]), int(high[0])]})
    return tuple(result)
