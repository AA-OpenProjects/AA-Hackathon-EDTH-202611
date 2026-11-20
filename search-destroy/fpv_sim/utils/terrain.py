import json
from importlib.resources import files

import pybullet as p

TERRAINS_DIR = files('fpv_sim') / 'assets' / 'terrains'
OBJECTS_DIR = files('fpv_sim') / 'assets' / 'objects'

def load_terrain(terrain: str, client: int):
    """Builds a terrain in the PyBullet world, with the target box at one of its listed positions.

    Parameters
    ----------
    terrain : str
        "<name>@v<N>": the scene `assets/terrains/<name>.json`, with the box at its N-th entry of `targets`.
    client : int
        The PyBullet client id.

    Returns
    -------
    list[int]
        The PyBullet body ids: the target box first, then the scene objects in file order.

    """
    name, sep, version = terrain.partition('@')
    if not sep or not version.startswith('v') or not version[1:].isdigit():
        raise ValueError("in load_terrain(), expected '<name>@v<N>', got '{:s}'".format(terrain))
    spec = json.loads((TERRAINS_DIR / (name + '.json')).read_text())
    index = int(version[1:]) - 1
    if not 0 <= index < len(spec['targets']):
        raise ValueError("in load_terrain(), {:s} has targets v1-v{:d}, got '{:s}'".format(name, len(spec['targets']), terrain))
    box = {'type': 'urdf', 'asset': 'target_box.urdf', **spec['targets'][index]}
    ids = []
    for obj in [box] + spec['objects']:
        if obj['type'] == 'urdf':
            ids.append(p.loadURDF(str(OBJECTS_DIR / obj['asset']),
                                  basePosition=obj['position_m'],
                                  baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                  useFixedBase=True, physicsClientId=client))
        elif obj['type'] == 'surface':
            half = [s / 2 for s in obj['size_m']]
            ids.append(p.createMultiBody(baseMass=0,    # static
                                         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client),
                                         baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=obj['rgba'], physicsClientId=client),
                                         basePosition=obj['position_m'],
                                         baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                         physicsClientId=client))
        else:
            raise ValueError("in load_terrain(), unknown object type '{:s}' in {:s}".format(obj['type'], terrain))
    return ids
