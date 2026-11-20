from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pybullet as p

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.utils.perception import Detector, IMU, VIO


def test_detector_and_depth_model_boundaries(monkeypatch):
    mask = np.array([[-1, 7, 7], [-1, -1, 7]])
    detector = Detector()

    class ZeroNoise:
        def random(self):
            return .5

        def normal(self, _, scale):
            return np.zeros_like(scale)

    assert detector.detect(mask, 8, 250) is None
    assert detector.detect(mask, 7, 250, ZeroNoise()) == (1, 0, 2, 1)
    assert detector.detect(mask, 7, 250.01, ZeroNoise()) is None
    assert detector.detect(mask, 8, 250, ZeroNoise()) is None

    class Dropped(ZeroNoise):
        def random(self):
            return .01

    assert detector.detect(mask, 7, 250, Dropped()) is None

    class Shifted(ZeroNoise):
        def normal(self, _, scale):
            return np.array([10, -10, -10, 10])

    assert detector.detect(mask, 7, 250, Shifted()) == (0, 0, 2, 1)

    env = ApproachAviary()
    try:
        depth = env.depth_model.predict(np.array([[0., 1.]]), ZeroNoise())
        np.testing.assert_allclose(depth, [[env.depth_camera.NEAR, 40.]])
        assert depth.dtype == np.float32
        assert env.depth_model.predict(np.array([[.5]])).shape == (1, 1)

        class OneSigma(ZeroNoise):
            def normal(self, _, scale):
                return scale

        distances = np.array([5., 10., 20.])
        raw = env.depth_camera.FAR * (1 - env.depth_camera.NEAR / distances) / (env.depth_camera.FAR - env.depth_camera.NEAR)
        noisy = env.depth_model.predict(raw, OneSigma())
        np.testing.assert_allclose(noisy, distances * 1.03, rtol=1e-5)
    finally:
        env.close()


def test_imu_adds_small_noise(monkeypatch):
    class FixedNoise:
        def normal(self, _, scale, size):
            return np.full(size, scale)

    monkeypatch.setattr("fpv_sim.utils.perception.np.random.normal", FixedNoise().normal)
    state = np.zeros(20)
    state[6] = 1  # identity orientation
    state[13:16] = [.1, -.2, .3]
    gyro, accel = IMU().read(state, 1 / 30)
    np.testing.assert_allclose(gyro, [.102, -.198, .302])
    np.testing.assert_allclose(accel, [.03, .03, 9.83])


def test_vio_accumulates_horizontal_error_and_resets():
    class OneSigma:
        def normal(self, _, scale, size=None):
            return np.full(size, scale) if size else scale

    state = np.zeros(20)
    state[:3] = [2, 3, 4]
    vio = VIO()
    vio.reset(state)
    state[:3] = [3, 3, 5]
    np.testing.assert_allclose(vio.read(state, OneSigma()), [3.013, 3.013, 5.05])
    np.testing.assert_allclose(vio.read(state, OneSigma()), [3.016, 3.016, 5.05])
    vio.reset(state)
    np.testing.assert_allclose(vio.read(state, OneSigma()), [3.003, 3.003, 5.05])


def test_full_stick_requests_sixty_degree_attitude(monkeypatch):
    env = ApproachAviary()
    try:
        target = {}
        def capture(**kwargs):
            target.update(kwargs)
            return np.zeros(4)
        monkeypatch.setattr(env.ctrl, "computeControlFromState", capture)
        env._preprocessAction([1, -1, 0, 0])
        assert np.allclose(target["target_rp"], np.radians([60, -60]))
    finally:
        env.close()


def test_sensor_api_and_safe_goal():
    env = ApproachAviary()
    try:
        obs, _ = env.reset(seed=1)
        visual = p.getVisualShapeData(env.DRONE_IDS[0], physicsClientId=env.CLIENT)[0]
        assert visual[3] == (1, 1, 1)
        mesh = Path(__file__).parents[1] / "fpv_sim/assets/drone/drone.dae"
        assert Path(visual[4].decode()).resolve() == mesh.resolve()
        scene = ElementTree.parse(mesh)
        ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
        transform = scene.find("./c:library_visual_scenes/c:visual_scene/c:node/c:matrix", ns)
        assert transform is not None and float(transform.text.split()[0]) == 0.00552
        assert env.observation_space.contains(obs)
        assert "prior_xy_m" not in obs
        assert "pose" not in obs
        assert obs["gyro_rad_s"].shape == obs["accel_m_s2"].shape == (3,)
        assert obs["gyro_rad_s"].dtype == obs["accel_m_s2"].dtype == np.float32
        assert np.isfinite(obs["gyro_rad_s"]).all() and np.isfinite(obs["accel_m_s2"]).all()
        np.testing.assert_allclose(obs["accel_m_s2"], [0, 0, 9.8], atol=.2)
        initial_gyro, initial_accel = obs["gyro_rad_s"].copy(), obs["accel_m_s2"].copy()
        repeated, _ = env.reset(seed=1)
        np.testing.assert_array_equal(repeated["gyro_rad_s"], initial_gyro)
        np.testing.assert_array_equal(repeated["accel_m_s2"], initial_accel)
        _, camera_seg = env.camera.read(env._getDroneStateVector(0), segmentation=True)
        assert not np.any(camera_seg == env.DRONE_IDS[0])
        assert obs["depth_m"].shape == (126, 224)
        assert obs["vio_xyz_m"].shape == (3,) and obs["vio_xyz_m"].dtype == np.float32
        assert np.max(obs["depth_m"]) <= 40
        raw_depth = env.depth_camera.read_depth_buffer(env._getDroneStateVector(0))
        depth = env.depth_model.predict(raw_depth, np.random.default_rng(1))
        assert raw_depth.shape == depth.shape == (126, 224) and depth.dtype == np.float32
        assert np.isfinite(depth).all() and np.all((0 <= depth) & (depth <= 40))
        assert not env._success()
        obs, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert not terminated and not truncated and not info["is_success"]
        # The 9 m around the start is clear in every terrain, so the box can sit at (3, 3) with nothing nearby.
        p.resetBasePositionAndOrientation(env.TERRAIN_IDS[0], [3, 3, 0], [0, 0, 0, 1], physicsClientId=env.CLIENT)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [5.49, 3, 1.55], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        env.seen = True
        assert np.isclose(env._distance(), .49)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [5.49, 3, 2.55], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert np.isclose(env._distance(), np.hypot(.49, 1.))
        assert not env._success() and not env._computeTerminated()
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [5.1, 3, 1], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], [3, 0, 0], [0, 0, 0], physicsClientId=env.CLIENT)
        p.performCollisionDetection(physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        env.seen = False
        assert env._success() and env._computeTerminated() and not env._collision()
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [5.5, 1.775, 1.55], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        p.performCollisionDetection(physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert np.isclose(env._distance(), np.hypot(.5, .4))
        assert not env._success()
        p.resetBasePositionAndOrientation(env.TERRAIN_IDS[0], [3, 3, 0],
                                          p.getQuaternionFromEuler([0, 0, np.pi / 4]),
                                          physicsClientId=env.CLIENT)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0],
                                          [3 + 2.5 / np.sqrt(2), 3 + 2.5 / np.sqrt(2), 1],
                                          [0, 0, 0, 1], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert np.isclose(env._distance(), .5)
    finally:
        env.close()


def test_floor_and_arena_bounds():
    env = ApproachAviary()
    try:
        env.reset(seed=1)
        floor = p.getVisualShapeData(env.PLANE_ID, physicsClientId=env.CLIENT)[0]
        assert floor[3] == (500, 500, .01)
        assert np.isclose(floor[5][2], -.055)
        assert p.getCollisionShapeData(env.PLANE_ID, -1, physicsClientId=env.CLIENT)[0][2] == p.GEOM_PLANE
        assert p.rayTest([300, 0, 1], [300, 0, -1], physicsClientId=env.CLIENT)[0][0] == env.PLANE_ID
        for xy, expected in [((349.9, 0), False), ((350, 0), True), ((0, -350), True)]:
            env.pos[0] = [*xy, 3]
            assert env._computeTerminated() == expected
    finally:
        env.close()
