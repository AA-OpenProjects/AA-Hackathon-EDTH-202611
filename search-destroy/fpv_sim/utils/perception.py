import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

class Camera:
    """Forward-facing onboard camera, defaults match the Inno U20 (stock 2.8 mm lens).

    Input: the drone state from `obs`. Output: an RGB frame in the drone's body
    frame, so it rolls and pitches with the airframe. Values come from PixelLock's
    `configs/cameras/inno_u20.json`: 1280x720, 102 deg HFOV, 30 fps.

    """

    ################################################################################

    def __init__(self,
                 client: int,
                 width: int=1280,
                 height: int=720,
                 hfov_deg: float=102.,
                 fps: int=30,
                 mount=(0.30, 0., 0.),
                 uptilt_deg: float=25.
                 ):
        """Camera __init__ method.

        Parameters
        ----------
        client : int
            The PyBullet client id, from env.getPyBulletClient().
        width, height : int, optional
            Image resolution in pixels.
        hfov_deg : float, optional
            Horizontal field of view in degrees.
        fps : int, optional
            Frame rate, read() it every round(ctrl_freq / fps) steps.
        mount : tuple, optional
            Camera position in the body frame (FLU), in metres, ahead of the front propeller sweep by default.
        uptilt_deg : float, optional
            Camera tilted up from the body x axis, in degrees: level flight at that pitch looks straight ahead.

        """
        self.CLIENT = client
        self.WIDTH, self.HEIGHT, self.FPS = width, height, fps
        self.MOUNT = np.array(mount)
        self.UPTILT_DEG = uptilt_deg
        #### Camera axes (forward, left, up) in the body frame: forward and up turn about body y, nose of the camera up
        t = np.radians(uptilt_deg)
        self.TILT = np.array([[np.cos(t), 0., -np.sin(t)],
                              [0., 1., 0.],
                              [np.sin(t), 0., np.cos(t)]])
        self.HFOV_DEG = hfov_deg
        #### PyBullet takes the vertical FOV; a pinhole with the U20's 102 deg HFOV at 16:9 gives ~70 deg
        # shortcut: pinhole, no lens distortion, so the profile's 55 deg VFOV (fisheye-ish lens) isn't matched; add a distortion remap if edge geometry matters
        self.VFOV_DEG = np.degrees(2 * np.arctan(np.tan(np.radians(hfov_deg) / 2) * height / width))
        self.NEAR = 0.02
        self.FAR = 1000.
        self.PROJECTION = p.computeProjectionMatrixFOV(fov=self.VFOV_DEG, aspect=width/height, nearVal=self.NEAR, farVal=self.FAR)
        #### DIRECT mode has no OpenGL
        self.RENDERER = p.ER_BULLET_HARDWARE_OPENGL if p.getConnectionInfo(client)['connectionMethod'] == p.GUI else p.ER_TINY_RENDERER

    ################################################################################

    def read(self, state, segmentation=False):
        """Renders a frame.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the drone state, from `obs`.
        segmentation : bool, optional
            Whether to also return the segmentation mask of the same render, for `Detector.detect`.

        Returns
        -------
        ndarray
            (height, width, 3)-shaped array of uint8's, RGB.
        ndarray
            Only if `segmentation`: (height, width)-shaped array of ints, the PyBullet body id seen in each pixel, -1 for none.

        """
        eye, rot = self._pose(state)
        view = p.computeViewMatrix(cameraEyePosition=eye,
                                   cameraTargetPosition=eye + rot[:, 0],    # camera forward, body x tilted up
                                   cameraUpVector=rot[:, 2],                # camera up, so the image rolls with the drone
                                   physicsClientId=self.CLIENT
                                   )
        w, h, rgba, _, seg = p.getCameraImage(width=self.WIDTH, height=self.HEIGHT, viewMatrix=view,
                                              projectionMatrix=self.PROJECTION, renderer=self.RENDERER,
                                              flags=0 if segmentation else p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.CLIENT)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
        return (rgb, np.reshape(seg, (h, w))) if segmentation else rgb

    ################################################################################

    def read_depth_buffer(self, state):
        """Return PyBullet's raw depth buffer, shape (height, width), values in [0, 1]."""
        eye, rot = self._pose(state)
        view = p.computeViewMatrix(eye, eye + rot[:, 0], rot[:, 2])
        _, _, _, depth, _ = p.getCameraImage(self.WIDTH, self.HEIGHT, viewMatrix=view,
                                             projectionMatrix=self.PROJECTION, renderer=self.RENDERER,
                                             physicsClientId=self.CLIENT)
        return np.asarray(depth).reshape(self.HEIGHT, self.WIDTH)

    ################################################################################

    def _pose(self, state):
        """The camera position and orientation (columns: forward, left, up) in the world."""
        rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        return state[0:3] + rot @ self.MOUNT, rot @ self.TILT

class Detector:
    """Simulator detector: segmentation, body ID, and range in; noisy pixel box or None out."""

    def detect(self, seg, body_id, distance_m, rng=None):
        """Return the visible body's inclusive (x1, y1, x2, y2) pixel box, or None.

        The mask comes from the same camera render, so occlusion and image-edge
        clipping are reflected in the box.

        Input: (height, width) segmentation mask, a PyBullet body ID, 3D distance
        to the body surface in metres, and an optional random generator.
        Output: pixel box or None when the body is not visible or farther than 250 m.
        """
        if distance_m > 250:
            return None
        rng = np.random if rng is None else rng
        ys, xs = np.nonzero(seg == body_id)
        if xs.size == 0 or rng.random() < .05:
            return None
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        width, height = box[2] - box[0] + 1, box[3] - box[1] + 1
        box = np.rint(box + rng.normal(0, [max(.5, .01 * width), max(.5, .01 * height)] * 2)).astype(int)
        box[::2] = np.clip(box[::2], 0, seg.shape[1] - 1)
        box[1::2] = np.clip(box[1::2], 0, seg.shape[0] - 1)
        return min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])


class DepthModel:
    """Raw PyBullet depth buffer in, noisy metric depth map out."""

    def __init__(self, camera):
        self.NEAR, self.FAR = camera.NEAR, camera.FAR
        self.MAX_DEPTH_M = 40.

    def predict(self, depth_buffer, rng=None):
        """Return float32 (height, width) depth in metres, clipped to [0, 40].

        Input: camera depth buffer with values in [0, 1], and an optional random generator.
        """
        rng = np.random if rng is None else rng
        depth_m = np.minimum(self.MAX_DEPTH_M,
                             self.FAR * self.NEAR / (self.FAR - (self.FAR - self.NEAR) * depth_buffer))
        return np.clip(depth_m + rng.normal(0, np.maximum(.03, .03 * depth_m)),
                       0, self.MAX_DEPTH_M).astype(np.float32)

################################################################################

def draw_box(rgb, box):
    """Return the frame with the detection box drawn on it, unchanged if `box` is None."""
    if box is None:
        return rgb
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(0, 255, 0), width=2)
    return np.asarray(img)

################################################################################

class IMU:
    """Noisy 6-axis IMU, gyroscope and accelerometer, at the body origin.

    Input: the simulator drone state, once per observation. Output: what the
    real chip measures, in the body frame (FLU): body rates, and specific force
    (acceleration minus gravity), so it reads [0, 0, +g] at rest and in hover.

    """

    ################################################################################

    def __init__(self, g: float=9.8):
        """IMU __init__ method.

        Parameters
        ----------
        g : float, optional
            The gravitational acceleration in m/s^2, as passed to the env.

        """
        # shortcut: white noise only, no bias drift or motor vibration; add those when testing estimators
        self.GRAVITY = np.array([0., 0., -g])
        self.prev_vel = None

    ################################################################################

    def reset(self):
        """Forgets the last velocity, call it after env.reset()."""
        self.prev_vel = None

    ################################################################################

    def read(self, state, dt, rng=None):
        """Reads the gyroscope and accelerometer.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped simulator drone state.
        dt : float
            Time since the previous read(), in seconds.
        rng : numpy random generator, optional
            Source of sensor noise.

        Returns
        -------
        ndarray
            (3,)-shaped array of floats, body rates in rad/s.
        ndarray
            (3,)-shaped array of floats, specific force in m/s^2.

        """
        rng = np.random if rng is None else rng
        rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        vel = np.array(state[10:13])
        #### First read has no previous velocity: assume unaccelerated
        acc = np.zeros(3) if self.prev_vel is None else (vel - self.prev_vel) / dt
        self.prev_vel = vel
        return (rot.T @ np.array(state[13:16]) + rng.normal(0, .002, 3),
                rot.T @ (acc - self.GRAVITY) + rng.normal(0, .03, 3))


class VIO:
    """Noisy local position estimate relative to the arena center.

    Input: successive drone states. Output: (x, y, altitude) in metres.
    Horizontal motion accumulates drift; altitude has independent measurement noise.
    """

    def reset(self, state):
        self.previous_xy = np.array(state[:2], dtype=float)
        self.estimated_xy = self.previous_xy.copy()

    def read(self, state, rng=None):
        rng = np.random if rng is None else rng
        xy = np.asarray(state[:2])
        delta = xy - self.previous_xy
        self.estimated_xy += delta + rng.normal(0, .003 + .01 * np.linalg.norm(delta), 2)
        self.previous_xy = xy.copy()
        return np.r_[self.estimated_xy, state[2] + rng.normal(0, .05)].astype(np.float32)
