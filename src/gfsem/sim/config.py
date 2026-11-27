"""All simulator parameters, as dataclasses.

Frames: world (track start at origin, x along the initial track direction, y left),
ego (origin at Juno's rear-axle centre, x forward, y left). Angles in radians
unless a field name says ``_deg``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VEHICLE_YAML = REPO_ROOT / "configs" / "vehicle.yaml"


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 1.60
    length: float = 2.875
    width: float = 1.165
    max_steer_deg: float = 10.0
    rear_overhang: float | None = None  # None -> symmetric overhangs
    max_steer_rate_deg: float = 40.0     # deg/s, actuator
    max_accel: float = 1.0               # m/s^2
    max_decel: float = 3.0               # m/s^2 (braking)

    @property
    def max_steer(self) -> float:
        return math.radians(self.max_steer_deg)

    @property
    def kappa_max(self) -> float:
        return math.tan(self.max_steer) / self.wheelbase

    @property
    def rear_oh(self) -> float:
        if self.rear_overhang is not None:
            return self.rear_overhang
        return (self.length - self.wheelbase) / 2.0

    @property
    def body_x(self) -> tuple[float, float]:
        """Body extent along x in the ego (rear-axle) frame."""
        return -self.rear_oh, self.length - self.rear_oh

    @classmethod
    def from_yaml(cls, path: Path = VEHICLE_YAML) -> "VehicleConfig":
        v = yaml.safe_load(path.read_text())["vehicle"]
        return cls(
            wheelbase=float(v["wheelbase_m"]),
            length=float(v["length_m"]),
            width=float(v["width_m"]),
            max_steer_deg=float(v.get("max_steer_deg", 10.0)),
            rear_overhang=v.get("rear_overhang_m"),
        )


@dataclass(frozen=True)
class ObstacleClass:
    radius: float
    height: float


def obstacle_classes_from_yaml(path: Path = VEHICLE_YAML) -> dict[str, ObstacleClass]:
    o = yaml.safe_load(path.read_text())["obstacles"]
    return {k: ObstacleClass(float(v["radius_m"]), float(v["height_m"])) for k, v in o.items()}


# Integer class ids used everywhere (labels, rasters, datasets).
BOLLARD, CONE = 1, 2


@dataclass(frozen=True)
class SceneConfig:
    track_width: float = 12.0
    kappa_max_track: float = 1.0 / 30.0
    p_straight: float = 0.4               # else constant curvature, |k| ~ U(0, kappa_max_track)
    track_back: float = 10.0              # track length behind the spawn [m]
    track_after_goal: float = 20.0        # track length past the goal line [m]
    field_length: float = 25.0
    field_start: tuple[float, float] = (8.0, 15.0)
    goal_margin: float = 2.0              # goal line this far past the end of the field
    n_bollards: tuple[int, int] = (6, 14)
    n_cones: tuple[int, int] = (0, 6)
    bollard_radius: float = 0.225
    bollard_height: float = 1.10
    cone_radius: float = 0.25
    cone_height: float = 0.40
    min_spacing: float = 1.0              # centre-to-centre, any pair
    edge_keepout: float = 0.3             # obstacle surface stays this far inside the track edge
    door_clearance: float = 0.25          # per side: door if gap - 2r >= width + 2 * clearance
    p_no_route: float = 0.05
    # Layout: with p_rows, bollards form 1-2 partial "rows" across the track with
    # gaps (creates real left/right choices); the rest are scattered uniformly.
    p_rows: float = 0.7
    row_spacing: tuple[float, float] = (1.1, 1.9)   # wall spacing (below the door threshold)
    row_min_gap_s: float = 8.0            # min distance between rows along the track
    max_resample: int = 60
    spawn_speed: float = 3.0
    target_speed: float = 3.0
    # Feasibility oracle (lattice search) -- defines "feasible" for scene labels.
    oracle_kappa_frac: float = 0.95       # use 95% of kappa_max
    oracle_inflation: float = 0.25        # obstacle inflation [m] = door_clearance (the door rule)
    oracle_step: float = 1.0              # primitive arc length [m]
    oracle_n_kappa: int = 7
    oracle_d_res: float = 0.2
    oracle_psi_res_deg: float = 2.0

    @classmethod
    def from_yaml(cls, path: Path = VEHICLE_YAML, **kw) -> "SceneConfig":
        oc = obstacle_classes_from_yaml(path)
        base = cls(
            bollard_radius=oc["bollard"].radius, bollard_height=oc["bollard"].height,
            cone_radius=oc["cone"].radius, cone_height=oc["cone"].height,
        )
        return replace(base, **kw)


@dataclass(frozen=True)
class LidarConfig:
    """RoboSense EMX at 1.10 m, pitched 7 deg down."""
    mount_x: float = 1.60                 # sensor position in the ego frame [m] (front axle)
    mount_y: float = 0.0
    mount_height: float = 1.10
    pitch_deg: float = 7.0                # down
    hfov_deg: float = 120.0
    h_step_deg: float = 0.2
    v_min_deg: float = -12.0              # around the pitched axis
    v_max_deg: float = 8.0
    blind_zone: float = 1.5
    max_range: float = 40.0
    edge_range: float = 25.0              # road-edge points (segmentation + ground intensity)
    edge_spacing: float = 0.5
    clutter_band: tuple[float, float] = (-0.3, 1.2)  # |d| - W/2 band where clutter appears
    # body profile below/ahead of the sensor in its vertical plane, (x from the rear axle, height) [m]; None = no
    # self-occlusion (all runs so far). Beams steeper than the profile's clearance angle hit the car.
    body_profile: tuple | None = None

    @property
    def n_rays(self) -> int:
        return int(round(self.hfov_deg / self.h_step_deg)) + 1

    @property
    def ray_angles(self):
        import numpy as np
        h = math.radians(self.hfov_deg) / 2
        return np.linspace(-h, h, self.n_rays)

    @property
    def elev_range(self) -> tuple[float, float]:
        """Absolute beam elevation range (radians, positive up); the lower bound is clipped to the body clearance
        angle when a body profile is set (beams below it hit the car)."""
        lo = math.radians(self.v_min_deg - self.pitch_deg)
        clr = self.body_clearance_rad
        return (max(lo, clr) if clr is not None else lo, math.radians(self.v_max_deg - self.pitch_deg))

    @property
    def body_clearance_rad(self) -> float | None:
        """Steepest downward beam elevation that still clears the body profile (radians), or None."""
        return body_clearance(self.mount_x, self.mount_height, self.body_profile)

    @classmethod
    def with_mount(cls, name: str, **kw) -> "LidarConfig":
        """A LiDAR config at a named mount (LIDAR_MOUNTS); other fields default or from ``kw``."""
        return cls(**{**LIDAR_MOUNTS[name], **kw})


def body_clearance(sx: float, sz: float, profile, n: int = 200) -> float | None:
    """Max elevation angle (radians) from the sensor (sx, sz) to the body polyline ``profile`` [(x, z), ...]
    ahead of it: beams below this angle hit the car."""
    if not profile:
        return None
    import numpy as np
    P = np.asarray(profile, float)
    best = -math.pi / 2
    for a, b in zip(P[:-1], P[1:]):
        t = np.linspace(0, 1, n)[:, None]
        Q = a + t * (b - a)
        ahead = Q[:, 0] > sx + 1e-6
        if ahead.any():
            best = max(best, float(np.max(np.arctan2(Q[ahead, 1] - sz, Q[ahead, 0] - sx))))
    return best


def placeholder_body_profile(sx: float, sz: float, body_front: float, edge_below: float = 0.035,
                             glass_len: float = 0.60, glass_slope_deg: float = 20.0, nose_top_z: float = 0.55):
    """ASSUMED windshield/hood profile until measured (DEVIATIONS): roof edge just ahead of / under the sensor
    (``edge_below`` m below the optical centre), windshield sloping ``glass_slope_deg`` below the horizontal over
    ``glass_len`` m (from the side photo), then a mid-hood point and the nose top at ``body_front``."""
    ze = sz - edge_below
    xb, zb = sx + glass_len, ze - glass_len * math.tan(math.radians(glass_slope_deg))
    xm, zm = 0.5 * (xb + body_front), 0.5 * (zb + nose_top_z) + 0.03
    return ((sx + 0.01, ze), (xb, zb), (xm, zm), (body_front, nose_top_z))


# LiDAR mounts: mount_x from the rear axle, optical-centre height, pitch down. "front_axle" = every run so far;
# "roof" = EMX on the roof at the base of the ZED support (top edge of the windshield), provisional until measured.
LIDAR_MOUNTS = {
    "front_axle": dict(mount_x=1.60, mount_height=1.10, pitch_deg=7.0),
    "roof": dict(mount_x=1.05, mount_height=1.20, pitch_deg=7.0,
                 body_profile=placeholder_body_profile(1.05, 1.20, 2.35)),
}


@dataclass(frozen=True)
class NoiseLevel:
    name: str
    range_sigma: float           # m
    angle_sigma_deg: float       # per-point bearing jitter
    dropout: float               # per-point drop probability
    clutter_rate: float          # mean false-positive clusters per scan
    edge_sigma: float            # m
    edge_dropout_rate: float     # mean 1-2 m edge dropouts per side per scan
    p_mis: float                 # class label flip probability
    drift_xy: float              # odom drift per replan, m (1 sigma)
    drift_yaw_deg: float         # odom drift per replan, deg (1 sigma)
    h_step_deg: float | None = None   # horizontal ray step override (None = LidarConfig.h_step_deg, 0.2 deg)


NOISE_LEVELS: dict[str, NoiseLevel] = {
    "L0": NoiseLevel("L0", 0.00, 0.00, 0.00, 0.0, 0.00, 0.0, 0.00, 0.000, 0.00),
    "L1": NoiseLevel("L1", 0.03, 0.05, 0.05, 0.5, 0.10, 0.3, 0.05, 0.005, 0.05),
    "L2": NoiseLevel("L2", 0.05, 0.10, 0.15, 1.5, 0.15, 0.6, 0.08, 0.010, 0.10),
    "L3": NoiseLevel("L3", 0.08, 0.20, 0.30, 3.0, 0.25, 1.0, 0.12, 0.020, 0.20),
    "L4": NoiseLevel("L4", 0.12, 0.30, 0.45, 5.0, 0.40, 1.5, 0.20, 0.040, 0.40),
    # Realistic RoboSense EMX (2026-11-26): range accuracy 5 cm @1 sigma and 0.08 deg horizontal resolution from the
    # EMX user guide (Table 1); dropout, clutter, edge noise, class flips, drift and bearing jitter are not in the
    # spec and use the L1 values (assumptions, DEVIATIONS). Not part of the L0-L4 ladder.
    "EMX": NoiseLevel("EMX", 0.05, 0.05, 0.05, 0.5, 0.10, 0.3, 0.05, 0.005, 0.05, h_step_deg=0.08),
    # beams-only comparison (2026-11-27): EMX beam pattern (0.08 deg) and range noise (0.05 m) with the L2 / L4
    # artefacts (bearing jitter, dropout, clutter, edge noise/dropouts, class flips, drift), vs L2 / L4 themselves.
    "emx_L2art": NoiseLevel("emx_L2art", 0.05, 0.10, 0.15, 1.5, 0.15, 0.6, 0.08, 0.010, 0.10, h_step_deg=0.08),
    "emx_L4art": NoiseLevel("emx_L4art", 0.05, 0.30, 0.45, 5.0, 0.40, 1.5, 0.20, 0.040, 0.40, h_step_deg=0.08),
}
LADDER = ("L0", "L1", "L2", "L3", "L4")        # the robustness / stress levels (EMX is separate)


@dataclass(frozen=True)
class ControllerConfig:
    lookahead: float = 2.5
    dt: float = 0.05             # simulation step [s]
    replan_every: int = 2        # replan at 1 / (dt * replan_every) = 10 Hz
    timeout: float = 40.0        # s
    stop_speed: float = 0.05     # m/s, below this the car counts as stopped
    stop_hold: float = 1.0       # s stopped -> episode ends
    # speed mode: "constant" (every result so far: the planner's speed, 3 m/s, accel/brake limits of the vehicle) or
    # "juno_bangbang" (Juno's real controller: ICE valve open below v_lo, closed above v_hi, coasting in between;
    # never brakes; the planner's brake command uses the real brake). Juno values are ASSUMED placeholders until
    # measured (DEVIATIONS 2026-11-27).
    speed_mode: str = "constant"
    juno: "JunoSpeedConfig" = field(default_factory=lambda: JunoSpeedConfig())


@dataclass(frozen=True)
class JunoSpeedConfig:
    v_lo: float = 1.67           # m/s (6 km/h): valve opens
    v_hi: float = 2.22           # m/s (8 km/h): valve closes
    a_open: float = 1.0          # m/s^2 valve open (assumed)
    d_coast: float = 0.1         # m/s^2 coasting, valve closed (assumed)
    valve_delay: float = 0.2     # s command to valve (assumed)
    d_brake: float = 3.0         # m/s^2 brake (assumed)
    brake_delay: float = 0.3     # s brake command to braking (assumed)

    def stopping_distance(self, v: float | None = None, latency: float = 0.0) -> float:
        """Distance from the brake decision to standstill from ``v`` (default v_hi): latency + brake delay at constant
        speed, then constant deceleration."""
        v = self.v_hi if v is None else v
        return v * (latency + self.brake_delay) + v * v / (2 * self.d_brake)


@dataclass(frozen=True)
class SimConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig.from_yaml)
    scene: SceneConfig = field(default_factory=SceneConfig.from_yaml)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    control: ControllerConfig = field(default_factory=ControllerConfig)
