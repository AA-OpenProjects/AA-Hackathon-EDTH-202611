"""Fly a terrain scene with a USB RC transmitter."""

import argparse
import json
import numpy as np
import sys

from fpv_sim.utils.rc_input import RCInput
from fpv_sim.utils.terrain import TERRAINS_DIR

import pygame


def rc_action(sticks, hover_throttle):
    """Map transmitter throttle [0, 1] to the environment's hover-centred stick."""
    roll, pitch, yaw, throttle = sticks
    return np.array([roll, pitch, yaw, np.clip(throttle / hover_throttle - 1, -1, 1)])


def main(terrain="village@v1"):
    try:
        rc = RCInput()
    except RuntimeError as exc:
        pygame.quit()
        print(exc, file=sys.stderr)
        return 1

    from fpv_sim.envs.approach_aviary import ApproachAviary
    from fpv_sim.utils.perception import Camera, Detector, draw_box
    from handin.agent import Agent

    env = ApproachAviary(continuous=True, terrain=terrain)
    try:
        camera = Camera(env.CLIENT, width=480, height=270)
        detector = Detector()
        window_size = (960, 540)
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Object reacquisition — RC flight")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 32)
        agent = Agent()
        obs, _ = env.reset()
        autonomous = False
        armed = False
        flying = False
        running = True
        fullscreen = False
        print(f"Lower throttle to arm, then raise it above {env.ctrl.HOVER_THROTTLE:.0%} to start. Set CH5 to 100 to hand control to the agent. Press R to reset, F11 for fullscreen; close the window to quit.")
        while running:
            sticks = rc.read()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, _ = env.reset()
                    autonomous = False
                    armed = False
                    flying = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    if fullscreen:
                        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
                    else:
                        window_size = screen.get_size()
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    fullscreen = not fullscreen
            if not running:
                break
            armed = armed or sticks[3] < .05
            flying = flying or (armed and sticks[3] >= env.ctrl.HOVER_THROTTLE)
            if rc.switch() and not autonomous:
                agent.reset()
            autonomous = rc.switch()
            if autonomous:
                obs, *_ = env.step(agent.act(obs))
            elif flying:
                obs, *_ = env.step(rc_action(sticks, env.ctrl.HOVER_THROTTLE))
            rgb, seg = camera.read(env._getDroneStateVector(0), segmentation=True)
            frame = draw_box(rgb, detector.detect(seg, env.TERRAIN_IDS[0], env._distance()))
            width, height = screen.get_size()
            scale = min(width / camera.WIDTH, height / camera.HEIGHT)
            size = (max(1, int(camera.WIDTH * scale)), max(1, int(camera.HEIGHT * scale)))
            screen.fill("black")
            screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), size),
                        ((width - size[0]) // 2, (height - size[1]) // 2))
            ground_speed = np.linalg.norm(env.vel[0][:2])
            screen.blit(font.render(f"Ground speed: {ground_speed:4.1f} m/s ({ground_speed * 3.6:3.0f} km/h)",
                                    True, "white", "black"), (10, 10))
            screen.blit(font.render(f"Altitude: {env.pos[0][2]:5.1f} m", True, "white", "black"), (10, 40))
            screen.blit(font.render("AGENT" if autonomous else "PILOT", True, "white", "black"), (10, 70))
            pygame.display.flip()
            clock.tick(30)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", default="village@v1",
                        choices=tuple(f"{name}@v{version}" for name in ("field", "forest", "village", "city")
                                      for version in range(1, 1 + len(json.loads((TERRAINS_DIR / f"{name}.json").read_text())["targets"]))))
    raise SystemExit(main(parser.parse_args().terrain))
