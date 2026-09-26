"""Solid-body physical properties in SI units; inertias about COM in link axes."""
from pathlib import Path
import math
import struct

import numpy as np


def finite_vector(value, size=3, name='vector'):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: expected finite vector') from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return result


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name}: expected positive finite value')
    return float(value)


def rotation(rpy):
    r, p, y = finite_vector(rpy, name='rpy')
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def transform(origin=None):
    origin = origin or {}
    return rotation(origin.get('rpy', [0, 0, 0])), finite_vector(origin.get('xyz', [0, 0, 0]))


def tensor6(matrix):
    a = np.asarray(matrix)
    return [float(a[0, 0]), float(a[1, 1]), float(a[2, 2]),
            float(a[0, 1]), float(a[0, 2]), float(a[1, 2])]


def tensor_matrix(values):
    xx, yy, zz, xy, xz, yz = finite_vector(values, 6, 'inertia')
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def mesh_scale(geometry):
    units = geometry.get('units')
    if units not in ('m', 'mm'):
        raise ValueError('mesh.units: explicitly specify m or mm')
    scale = finite_vector(geometry.get('scale', [1, 1, 1]), name='mesh.scale')
    if np.any(scale <= 0):
        raise ValueError('mesh.scale: expected positive components')
    return scale * (0.001 if units == 'mm' else 1)


def read_stl(path):
    """Read binary or ASCII STL without treating its facet normals as authoritative."""
    try:
        raw = Path(path).read_bytes()
        count = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else -1
        if len(raw) == 84 + 50 * count:
            triangles = [struct.unpack_from('<9f', raw, 96 + 50 * i) for i in range(count)]
            result = np.asarray(triangles).reshape(-1, 3, 3)
        else:
            vertices = [list(map(float, line.split()[1:])) for line in raw.decode('ascii').splitlines()
                        if line.strip().startswith('vertex ')]
            result = np.asarray(vertices).reshape(-1, 3, 3)
    except (OSError, ValueError, UnicodeError, struct.error) as exc:
        raise ValueError(f'{path}: invalid STL: {exc}') from exc
    if len(result) < 4 or not np.isfinite(result).all():
        raise ValueError(f'{path}: STL requires finite closed triangles')
    return result


def _solid_mesh(geometry):
    triangles = read_stl(geometry['path']) * mesh_scale(geometry)
    edges = {}
    for triangle in triangles:
        if np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])) <= 0:
            raise ValueError('STL has degenerate triangles')
        vertices = [tuple(vertex) for vertex in triangle]
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            key = tuple(sorted((a, b)))
            edges[key] = edges.get(key, ()) + ((a, b),)
    if any(len(pair) != 2 or pair[0] != pair[1][::-1] for pair in edges.values()):
        raise ValueError('STL must be watertight with consistent outward winding')
    # Shift near the mesh to avoid cancellation in translated CAD coordinates.
    anchor = np.mean(triangles.reshape(-1, 3), axis=0)
    local = triangles - anchor
    volumes = np.einsum('ij,ij->i', local[:, 0], np.cross(local[:, 1], local[:, 2])) / 6
    volume = float(volumes.sum())
    if volume <= 0:
        raise ValueError('STL must have positive volume and outward winding')
    sums = local.sum(axis=1)
    com = np.einsum('i,ij->j', volumes, sums) / (4 * volume)
    second = sum(v * (np.outer(s, s) + t.T @ t) / 20 for v, s, t in zip(volumes, sums, local))
    central = second - volume * np.outer(com, com)
    inertia = np.trace(central) * np.eye(3) - central
    if np.linalg.eigvalsh(inertia)[0] <= 0:
        raise ValueError('STL has invalid solid inertia')
    area = np.linalg.norm(np.cross(local[:, 1]-local[:, 0], local[:, 2]-local[:, 0]), axis=1).sum()/2
    return volume, com + anchor, inertia, float(area)


def geometry_properties(geometry, density):
    """Exact homogeneous solid integrals; STL must be closed, oriented, non-intersecting.

    Self intersections cannot be certified by edge checks and must be repaired in CAD.
    Returned inertia is a 3x3 matrix about COM, after the geometry origin transform.
    """
    density = positive(density, 'material.density')
    kind = geometry.get('type')
    com = np.zeros(3)
    if kind == 'box':
        size = finite_vector(geometry.get('size'), name='box.size')
        if np.any(size <= 0):
            raise ValueError('box.size: expected positive dimensions')
        x, y, z = size
        volume, area = float(np.prod(size)), 2*(x*y+x*z+y*z)
        inertia = np.diag([y*y+z*z, x*x+z*z, x*x+y*y]) * volume/12
    elif kind in ('sphere', 'cylinder'):
        r = positive(geometry.get('radius'), kind + '.radius')
        if kind == 'sphere':
            volume, area = 4*math.pi*r**3/3, 4*math.pi*r*r
            inertia = np.eye(3) * (2*volume*r*r/5)
        else:
            length = positive(geometry.get('length'), 'cylinder.length')
            volume, area = math.pi*r*r*length, 2*math.pi*r*(r+length)
            inertia = np.diag([(3*r*r+length*length)/12]*2 + [r*r/2]) * volume
    elif kind == 'mesh':
        volume, com, inertia, area = _solid_mesh(geometry)
    else:
        raise ValueError('geometry.type: expected box, sphere, cylinder or mesh')
    rot, pos = transform(geometry.get('origin'))
    return {'volume': float(volume), 'surface_area': float(area), 'mass': volume*density,
            'density': density, 'com': (pos + rot @ com).tolist(),
            'inertia': (rot @ (inertia*density) @ rot.T).tolist()}
