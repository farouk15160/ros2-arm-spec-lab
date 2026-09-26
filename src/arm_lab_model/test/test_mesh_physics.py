import math
import numpy as np
import pytest

from arm_lab_model.mesh_physics import geometry_properties


def test_box_rotated_translated_keeps_com_tensor_in_link_axes():
    props = geometry_properties({'type': 'box', 'size': [2, 4, 6],
        'origin': {'xyz': [3, 5, 7], 'rpy': [0, 0, math.pi / 2]}}, 10)
    assert props['volume'] == pytest.approx(48)
    assert props['mass'] == pytest.approx(480)
    assert props['com'] == pytest.approx([3, 5, 7])
    assert np.array(props['inertia']) == pytest.approx(np.diag([1600, 2080, 800]))


def cube_stl(path):
    points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    text = 'solid cube\n' + ''.join('facet normal 0 0 0\nouter loop\n' + ''.join(
        'vertex ' + ' '.join(map(str, points[i])) + '\n' for i in face)
        + 'endloop\nendfacet\n' for face in faces) + 'endsolid cube\n'
    path.write_text(text)
    return path


def test_stl_cube_matches_analytic_solid_and_units(tmp_path):
    path = cube_stl(tmp_path/'cube.stl')
    p = geometry_properties({'type':'mesh','path':str(path),'units':'m'}, 2700)
    assert p['volume'] == pytest.approx(1)
    assert p['mass'] == pytest.approx(2700)
    assert p['com'] == pytest.approx([.5,.5,.5])
    assert np.array(p['inertia']) == pytest.approx(np.eye(3)*450)
    mm = geometry_properties({'type':'mesh','path':str(path),'units':'mm'}, 2700)
    assert mm['mass'] == pytest.approx(2.7e-6)


def test_open_mesh_rejected(tmp_path):
    path = cube_stl(tmp_path/'cube.stl')
    text = path.read_text()
    path.write_text('solid cube\n' + text[text.index('endfacet')+9:])
    with pytest.raises(ValueError, match='watertight'):
        geometry_properties({'type':'mesh','path':str(path),'units':'m'}, 2700)


@pytest.mark.parametrize('density', [0, -1, float('nan'), True])
def test_invalid_density_rejected(density):
    with pytest.raises(ValueError, match='density'):
        geometry_properties({'type':'box','size':[1,1,1]}, density)


def test_binary_stl_matches_ascii(tmp_path):
    import struct
    from arm_lab_model.mesh_physics import read_stl
    ascii_path = cube_stl(tmp_path/'ascii.stl')
    triangles = read_stl(ascii_path)
    binary = bytes(80) + struct.pack('<I',len(triangles)) + b''.join(
        struct.pack('<12fH',0,0,0,*triangle.ravel(),0) for triangle in triangles)
    path = tmp_path/'binary.stl'
    path.write_bytes(binary)
    props = geometry_properties({'type':'mesh','path':str(path),'units':'m'}, 1)
    assert props['volume'] == pytest.approx(1)
    assert props['surface_area'] == pytest.approx(6)


def test_nonuniform_stl_scale_rotates_full_tensor(tmp_path):
    path = cube_stl(tmp_path/'cube.stl')
    props = geometry_properties({'type':'mesh','path':str(path),'units':'m',
        'scale':[2,4,6],'origin':{'xyz':[10,20,30],'rpy':[0,0,math.pi/4]}},10)
    assert props['mass'] == pytest.approx(480)
    assert props['com'] == pytest.approx([10-1/math.sqrt(2),20+3/math.sqrt(2),33])
    assert np.array(props['inertia']) == pytest.approx(np.array([[1840,240,0],[240,1840,0],[0,0,800]]))


@pytest.mark.parametrize('geometry', [
    {'type':'mesh','path':'missing.stl','units':'cm'},
    {'type':'mesh','path':'missing.stl','units':'m','scale':[-1,1,1]},
    {'type':'mesh','path':'missing.stl','units':'m'},
    {'type':'box','size':[0,1,1]}, {'type':'box','size':[1,2]},
    {'type':'sphere','radius':-1}, {'type':'cylinder','radius':1,'length':0},
    {'type':'unknown'},
])
def test_invalid_geometry_rejected(geometry):
    with pytest.raises(ValueError):
        geometry_properties(geometry,1)


def test_sphere_and_cylinder_analytic_properties():
    sphere = geometry_properties({'type':'sphere','radius':1},3/(4*math.pi))
    assert sphere['mass'] == pytest.approx(1)
    assert np.array(sphere['inertia']) == pytest.approx(np.eye(3)*.4)
    cylinder = geometry_properties({'type':'cylinder','radius':1,'length':2},1/(2*math.pi))
    assert cylinder['mass'] == pytest.approx(1)
    assert np.array(cylinder['inertia']) == pytest.approx(np.diag([7/12,7/12,.5]))
