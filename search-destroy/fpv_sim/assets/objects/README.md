# Low polygon terrain objects

Each URDF is a static object with its origin at ground level and its front
facing −y. Dimensions and positions are in metres. Visuals are faceted OBJ
meshes in `meshes/`, one per colour, so PyBullet's renderer draws few
triangles per object. Collisions are convex hulls of the main parts. Load an
object into a PyBullet world with:

```python
from importlib.resources import files
import pybullet as p

path = files('fpv_sim') / 'assets' / 'objects' / 'tree.urdf'
body_id = p.loadURDF(str(path), basePosition=(x, y, 0),
                     useFixedBase=True, physicsClientId=client)
```

| Asset | Visible triangles | Collision hulls |
| --- | ---: | ---: |
| `tree` | 53 | 3 |
| `tree_02` (large oak) | 73 | 4 |
| `pine` | 46 | 4 |
| `bush` | 40 | 2 |
| `rock` | 37 | 2 |
| `hay_bale` | 28 | 1 |
| `shed` | 28 | 2 |
| `fence` (4 m) | 54 | 5 |
| `electric_pole` | 70 | 3 |
| `house_01` (two storeys) | 140 | 4 |
| `house_02` (bungalow, porch) | 144 | 7 |
| `house_03` (garage) | 140 | 6 |
| `building_01` (office) | 118 | 3 |
| `building_02` (shop) | 118 | 3 |
| `building_03` (apartments) | 286 | 8 |
| `building_04` (brick, shopfront) | 146 | 3 |
| `car_01`, `car_02`, `car_03` | 82 each | 2 each |
| `target_box` | 12 | 1 |

Windows, doors, lights, and trim are visual details without collision.
Broadleaf canopies start at least 1.95 m above the ground, so the 1.55 m target
box can sit under them without touching them.

[`tools/generate_assets.py`](../../../tools/generate_assets.py) generates
every file here except `target_box.urdf`. Its [contact sheet](previews/contact_sheet.png)
and the PNGs in `previews/` show each object rendered with PyBullet's
TinyRenderer. The [terrain scenes](../terrains/README.md) use these objects.
