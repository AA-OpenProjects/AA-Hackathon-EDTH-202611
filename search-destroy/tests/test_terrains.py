import json
import math
from pathlib import Path

import pybullet as p
import pytest

from fpv_sim.utils.terrain import load_terrain

TERRAINS = Path(__file__).parents[1] / 'fpv_sim/assets/terrains'
SCENES = ('field', 'forest', 'village', 'city')


@pytest.mark.parametrize('name', SCENES)
def test_scene_lists_distinct_spread_targets(name):
    spec = json.loads((TERRAINS / f'{name}.json').read_text())
    positions = [tuple(t['position_m']) for t in spec['targets']]
    assert 15 <= len(positions) <= 50
    assert all(math.dist(a[:2], b[:2]) >= 10 for i, a in enumerate(positions) for b in positions[i + 1:])
    assert all(t['context'] for t in spec['targets'])
    assert all(z == 0 for _, _, z in positions)
    assert max(x for x, _, _ in positions) - min(x for x, _, _ in positions) >= 40
    assert max(y for _, y, _ in positions) - min(y for _, y, _ in positions) >= 30
    surfaces = [obj for obj in spec['objects'] if obj['type'] == 'surface']
    assert surfaces[0]['size_m'][:2] == [500, 500] and surfaces[0]['position_m'][:2] == [0, 0]
    props = [obj for obj in spec['objects'] if obj['type'] == 'urdf']
    assert spec['objects'] == props + surfaces
    near_target = lambda x, y: any(abs(x - tx) < 20 and abs(y - ty) < 20 for tx, ty, _ in positions)
    assert all(max(abs(x), abs(y)) < 150 or near_target(x, y) for x, y, _ in (obj['position_m'] for obj in props))
    if name == 'city':
        assert all(max(map(abs, obj['position_m'][:2])) < 150 for obj in props if obj['asset'].startswith('building_'))


@pytest.mark.parametrize('name', SCENES)
def test_targets_do_not_clip_and_start_is_clear(name):
    spec = json.loads((TERRAINS / f'{name}.json').read_text())
    client = p.connect(p.DIRECT)
    try:
        ids = load_terrain(f'{name}@v1', client)
        assert len(ids) == 1 + len(spec['objects'])
        assert all(p.getDynamicsInfo(body, -1, physicsClientId=client)[0] == 0 for body in ids)
        solid = [body for obj, body in zip(spec['objects'], ids[1:])
                 if obj['type'] != 'surface' and obj['asset'] != 'bush.urdf']
        for target in spec['targets']:
            p.resetBasePositionAndOrientation(ids[0], target['position_m'], p.getQuaternionFromEuler(target['rpy_rad']),
                                              physicsClientId=client)
            p.performCollisionDetection(physicsClientId=client)
            assert p.getCollisionShapeData(ids[0], -1, physicsClientId=client)[0][3] == (4, 1.65, 1.55)
            assert not any(p.getClosestPoints(ids[0], body, 0, physicsClientId=client) for body in solid), target
        drone = p.createMultiBody(baseMass=0, physicsClientId=client, basePosition=[0, 0, 3],
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=2, physicsClientId=client))
        p.performCollisionDetection(physicsClientId=client)
        assert not any(p.getClosestPoints(drone, body, 0, physicsClientId=client) for body in ids)
    finally:
        p.disconnect(client)


@pytest.mark.parametrize('version', [1, 6, 7, 40])
def test_version_selects_target(version):
    spec = json.loads((TERRAINS / 'village.json').read_text())
    client = p.connect(p.DIRECT)
    try:
        ids = load_terrain(f'village@v{version}', client)
        position = p.getBasePositionAndOrientation(ids[0], physicsClientId=client)[0]
        assert position == pytest.approx(spec['targets'][version - 1]['position_m'], abs=1e-6)
    finally:
        p.disconnect(client)


@pytest.mark.parametrize('terrain', ['village', 'village@1', 'village@v41', 'village@v0'])
def test_bad_terrain_names_raise(terrain):
    client = p.connect(p.DIRECT)
    try:
        with pytest.raises(ValueError):
            load_terrain(terrain, client)
    finally:
        p.disconnect(client)
