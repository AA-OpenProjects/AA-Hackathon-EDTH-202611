import xml.etree.ElementTree as etxml

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

from fpv_sim import DRONE_URDF

class AngleControl(BaseControl):
    """Angle mode (self-levelling) control class for `fpv_sim.DRONE_URDF`, flown as DroneModel.RACE.

    Takes RC-style inputs, throttle in [0, 1], target roll and pitch angles
    in rad, and a target yaw rate in rad/s, as in Betaflight's angle mode.
    Passing `target_rates` instead skips the angle loop, as in acro mode.
    A cascaded P angle loop feeds a P body-rate loop whose torques are
    scaled by the URDF inertia, then the motor allocation is inverted
    into RPMs through `kf` and `km`.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Angle control class __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.RACE:
            raise ValueError("in AngleControl.__init__(), AngleControl requires DroneModel.RACE")
        self.J = np.array([self._getURDFParameter(i) for i in ['ixx', 'iyy', 'izz']])
        self.MAX_THRUST = self._getURDFParameter('thrust2weight') * self.GRAVITY
        self.MAX_RPM = np.sqrt(self.MAX_THRUST / (4*self.KF))
        self.HOVER_THROTTLE = self.GRAVITY / self.MAX_THRUST
        self.P_COEFF_ANGLE = np.array([6., 6.])         # 1/s, angle error -> roll/pitch rate setpoint
        self.P_COEFF_RATE = np.array([30., 30., 10.])   # 1/s, rate error -> angular acceleration
        #### Allocation: [thrust, x_torque, y_torque, z_torque] = A @ rpm**2
        #### Signs follow BaseAviary._physics() for DroneModel.RACE, levers are the URDF prop positions:
        #### Read lever arms from the URDF so allocation matches the drone geometry.
        xy = np.array([[float(v) for v in etxml.parse(DRONE_URDF).getroot().find("link[@name='prop{:d}_link']/inertial/origin".format(i)).attrib['xyz'].split()[:2]] for i in range(4)])
        A = np.array([[1, 1, 1, 1],
                      xy[:, 1],
                      -xy[:, 0],
                      [self.KM/self.KF, -self.KM/self.KF, self.KM/self.KF, -self.KM/self.KF]
                      ]) * self.KF
        self.A_INV = np.linalg.inv(A)

    ################################################################################

    def _getURDFParameter(self, parameter_name: str):
        """BaseControl's URDF reader, on `fpv_sim.DRONE_URDF` instead of upstream's racer.urdf."""
        root = etxml.parse(DRONE_URDF).getroot()
        inertial = root.find('link/inertial')
        if parameter_name == 'm':
            return float(inertial.find('mass').attrib['value'])
        if parameter_name in ['ixx', 'iyy', 'izz']:
            return float(inertial.find('inertia').attrib[parameter_name])
        return float(root.find('properties').attrib[parameter_name])

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                throttle,
                                target_rp=np.zeros(2),
                                target_yaw_rate=0.,
                                target_rates=None
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        throttle : float
            Collective throttle in [0, 1], fraction of the maximum thrust.
        target_rp : ndarray, optional
            (2,1)-shaped array of floats containing the desired roll and pitch, in rad.
        target_yaw_rate : float, optional
            The desired yaw rate, in rad/s.
        target_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired body rates, in rad/s (acro mode).
            When given, `target_rp` and `target_yaw_rate` are ignored.

        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_quat=state[3:7],
                                   cur_ang_vel=state[13:16],
                                   throttle=throttle,
                                   target_rp=target_rp,
                                   target_yaw_rate=target_yaw_rate,
                                   target_rates=target_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_quat,
                       cur_ang_vel,
                       throttle,
                       target_rp=np.zeros(2),
                       target_yaw_rate=0.,
                       target_rates=None
                       ):
        """Computes the angle mode control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed (unused, the loops are proportional).
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity, in the world frame.
        throttle : float
            Collective throttle in [0, 1], fraction of the maximum thrust.
        target_rp : ndarray, optional
            (2,1)-shaped array of floats containing the desired roll and pitch, in rad.
        target_yaw_rate : float, optional
            The desired yaw rate, in rad/s.
        target_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired body rates, in rad/s (acro mode).
            When given, `target_rp` and `target_yaw_rate` are ignored.

        Returns
        -------
        ndarray
            (4,1)-shaped array of floats containing the RPMs to apply to each of the 4 motors.

        """
        self.control_counter += 1
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        body_rates = cur_rotation.T @ np.array(cur_ang_vel)
        #### Angle loop -> rate setpoint (skipped in acro), rate loop -> torques
        if target_rates is None:
            target_rates = np.append(self.P_COEFF_ANGLE * (np.array(target_rp) - cur_rpy[:2]), target_yaw_rate)
        torques = self.J * self.P_COEFF_RATE * (target_rates - body_rates) \
                  + np.cross(body_rates, self.J * body_rates)
        thrust = np.clip(throttle, 0., 1.) * self.MAX_THRUST
        #### Yaw gets what thrust, roll and pitch leave, as in Betaflight's mixer: prop drag gives ~0.01 Nm of yaw at hover,
        #### so a full yaw request would cut a motor pair to 0 RPM and take roll/pitch authority with it
        base = self.A_INV @ np.append(thrust, [*torques[:2], 0.])
        yaw = self.A_INV[:, 3] * torques[2]
        idle = .05 * self.MAX_RPM**2    # Betaflight's motor idle: 5% thrust, yaw alone never stops a motor
        room = np.where(yaw < 0, base - idle, self.MAX_RPM**2 - base) / np.maximum(np.abs(yaw), 1e-12)
        rpm_sq = base + np.clip(room.min(), 0., 1.) * yaw
        return np.clip(np.sqrt(np.clip(rpm_sq, 0., None)), 0., self.MAX_RPM)
