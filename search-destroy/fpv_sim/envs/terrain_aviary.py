import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from fpv_sim import DRONE_URDF
from fpv_sim.utils.terrain import load_terrain

class TerrainAviary(CtrlAviary):
    """CtrlAviary flying `fpv_sim.DRONE_URDF` in a versioned terrain, with quadratic body drag in place of PyBullet's damping.

    The plant can be perturbed away from the URDF, for domain randomisation, through attributes reset to nominal on
    every reset: THRUST_SCALE (per motor, on kf), KM_SCALE, COG (body x/y shift of the CoG, m), BODY_DRAG and WIND (world, m/s).

    """

    TOP_SPEED_KMH = 119.5
    TOP_SPEED_PITCH_DEG = 60.

    ################################################################################

    def __init__(self,
                 terrain: str=None,
                 obstacles=False,
                 **kwargs
                 ):
        """Initialization of an aviary environment with an optional terrain.

        Parameters
        ----------
        terrain : str, optional
            A versioned terrain, "<name>@v<N>" from `fpv_sim/assets/terrains`, replaces the default obstacles.
        obstacles : bool, optional
            Whether to add BaseAviary's default obstacles, when no terrain is given.
        **kwargs
            Passed to CtrlAviary.

        """
        self.TERRAIN = terrain
        super().__init__(obstacles=obstacles or terrain is not None, **kwargs)

    ################################################################################

    def _parseURDFParameters(self):
        """Points BaseAviary at the packaged drone URDF; an absolute path wins over its `assets/` join, here and in loadURDF."""
        self.URDF = str(DRONE_URDF)
        return super()._parseURDFParameters()

    ################################################################################

    def _addObstacles(self):
        """Adds the terrain, or BaseAviary's default obstacles if none was given.

        Called on every reset, after the world is wiped.

        """
        if self.TERRAIN is None:
            super()._addObstacles()
        else:
            self.TERRAIN_IDS = load_terrain(self.TERRAIN, self.CLIENT)

    ################################################################################

    def _housekeeping(self):
        """Swaps PyBullet's default linear damping (~0.8/s, a 55 km/h ceiling at 45 deg) for the body drag below.

        Called on every reset, after the drones are loaded.

        """
        super()._housekeeping()
        p.removeBody(self.PLANE_ID, physicsClientId=self.CLIENT)
        self.PLANE_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.CLIENT),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[250, 250, .005],
                visualFramePosition=[0, 0, -.055 if self.TERRAIN else -.005],
                rgbaColor=[1, 1, 1, 1], physicsClientId=self.CLIENT),
            physicsClientId=self.CLIENT)
        for i in range(self.NUM_DRONES):
            p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0., physicsClientId=self.CLIENT)
        #### Level flight at top speed: drag balances the horizontal thrust, m g tan(pitch)
        self.BODY_DRAG = self.M * self.G * np.tan(np.radians(self.TOP_SPEED_PITCH_DEG)) / (self.TOP_SPEED_KMH / 3.6)**2
        self.THRUST_SCALE, self.KM_SCALE, self.COG, self.WIND = np.ones(4), 1., np.zeros(2), np.zeros(3)

    ################################################################################

    def _physics(self, rpm, nth_drone):
        """BaseAviary's motor forces and torques with the plant's perturbations, plus the body drag on airspeed, -BODY_DRAG |v - WIND| (v - WIND)."""
        forces = rpm**2 * self.KF * self.THRUST_SCALE
        torques = -rpm**2 * self.KM * self.KM_SCALE * self.THRUST_SCALE    # minus: DroneModel.RACE spins the other way
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone], i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT)
        #### A CoG shifted by COG puts the thrust off-centre: -COG x thrust about it.
        # shortcut: torque only, gravity and the inertia stay at the URDF CoG; move the inertial frame if offsets grow past a few cm
        thrust = forces.sum()
        p.applyExternalTorque(self.DRONE_IDS[nth_drone], 4, flags=p.LINK_FRAME, physicsClientId=self.CLIENT,
                              torqueObj=[-self.COG[1] * thrust, self.COG[0] * thrust, -torques[0] + torques[1] - torques[2] + torques[3]])
        # shortcut: same drag in every direction, whatever the attitude; make it attitude-dependent if climbs or sideways flight matter
        vel = self.vel[nth_drone] - self.WIND
        p.applyExternalForce(self.DRONE_IDS[nth_drone], -1, forceObj=-self.BODY_DRAG * np.linalg.norm(vel) * vel,
                             posObj=self.pos[nth_drone], flags=p.WORLD_FRAME, physicsClientId=self.CLIENT)
