import json
from pathlib import Path

import numpy as np
import pygame
import pybullet as p

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.utils.rc_input import RCInput
from fpv_sim.scripts.rc import main, rc_action


def test_missing_radio_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame, "quit", lambda: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 0)
    assert main() == 1
    assert capsys.readouterr().err == "No USB joystick at index 0. Connect the radio in USB Joystick mode, then retry.\n"


def test_radio_mapping_and_hover_conversion(monkeypatch):
    class Joystick:
        def get_axis(self, axis):
            return [0.5, 0.01, -1., -0.3, 1.][axis]

    monkeypatch.setattr(pygame, "init", lambda: None)
    monkeypatch.setattr(pygame.event, "pump", lambda: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 1)
    monkeypatch.setattr(pygame.joystick, "Joystick", lambda _: Joystick())
    rc = RCInput(invert=(False, False, True, False))
    assert np.allclose(rc.read(), [.5, 0, .3, 0])
    assert rc.switch()
    assert np.allclose(rc_action(rc.read(), .4), [.5, 0, .3, -1])
    assert np.allclose(rc_action([0, 0, 0, .4], .4), [0, 0, 0, 0])
    assert np.allclose(rc_action([0, 0, 0, 1], .4), [0, 0, 0, 1])


def test_manual_flight_continues_after_task_limits():
    env = ApproachAviary(continuous=True)
    try:
        env.reset(seed=1)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [0, 0, .1], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert env._collision()
        _, _, terminated, truncated, info = env.step([0, 0, 0, 0])
        assert info["collision"]
        assert not terminated and not truncated
        env.step_counter = 240 * 60
        assert not env._computeTruncated()
        # The 9 m around the start is clear in every terrain, so the box can sit at (3, 3) with nothing nearby.
        p.resetBasePositionAndOrientation(env.TERRAIN_IDS[0], [3, 3, 0], [0, 0, 0, 1], physicsClientId=env.CLIENT)
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [5.1, 3, 1], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=env.CLIENT)
        p.performCollisionDetection(physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert env._success()
        assert not env._computeTerminated()
        _, _, terminated, truncated, info = env.step([0, 0, 0, 0])
        assert info["is_success"]
        assert not terminated and not truncated
    finally:
        env.close()


def test_rc_environment_can_use_forest_terrain():
    env = ApproachAviary(continuous=True, terrain="forest@v1")
    try:
        env.reset()
        spec = json.loads((Path(__file__).parents[1] / 'fpv_sim/assets/terrains/forest.json').read_text())
        assert len(env.TERRAIN_IDS) == 1 + len(spec['objects'])
        x, y, _ = spec['targets'][0]['position_m']
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [x, y, 2], [0, 0, 0, 1],
                                          physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        assert 0 < env._distance() < 1.5
    finally:
        env.close()
