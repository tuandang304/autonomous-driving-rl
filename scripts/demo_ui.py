"""
Game-style visual demo for trained agents.

A custom pygame renderer draws the highway-env simulation with procedural car
sprites, scenery, day/night lighting, effects and a HUD with the agent's live
Q-values. The renderer only *reads* simulation state: environment config,
seeding, observations and action selection are identical to evaluate.py, so the
agent behaves exactly as it does during evaluation.

Usage:
    python scripts/demo_ui.py
    python scripts/demo_ui.py --algorithm dueling_dqn \
        --weights weights/final/dueling_dqn_step50000.pth \
        --lanes 4 --density 2 --seed 6 --episodes 1

Controls:
    SPACE pause / continue   S sensor overlay   T day / night theme
    R restart episode        N next episode     ESC quit
"""

import argparse
import math
import os
import random
import sys
from collections import deque
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401  (registers highway-v0)
import numpy as np
import pygame
import pygame.gfxdraw
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import ActionSmoother, load_agent  # noqa: E402
from src.env.endless_highway import make_long_episode  # noqa: E402
from src.utils import set_seed  # noqa: E402


ACTION_LABELS = ("LANE LEFT", "IDLE", "LANE RIGHT", "FASTER", "SLOWER")
ALGO_LABELS = {"dqn": "DQN", "double_dqn": "DOUBLE DQN", "dueling_dqn": "DUELING DQN",
               "d3qn": "D3QN", "ppo": "PPO", "rule_based": "RULE-BASED"}
CAR_KINDS = ("sedan", "hatch", "van", "pickup")
TRAFFIC_COLORS = [(214, 58, 58), (38, 104, 196), (236, 236, 232), (34, 36, 42), (150, 156, 168),
                  (246, 186, 48), (58, 150, 92), (122, 72, 168), (204, 112, 44), (90, 180, 210)]

THEMES = {
    "day": {
        "grass": (88, 148, 64), "grass_alt": (81, 139, 59), "edge_dirt": (104, 112, 74),
        "asphalt": (64, 66, 72), "track": (57, 59, 64), "shoulder": (82, 84, 90),
        "line": (238, 238, 230), "rail": (200, 204, 210), "post": (112, 114, 120),
        "ambient": None,
        "accent": (255, 184, 28), "accent2": (255, 92, 56), "good": (80, 220, 140),
        "warn": (255, 170, 40), "bad": (255, 72, 72), "sensor": (40, 200, 255),
        "hud_top": (26, 29, 38), "hud_bot": (14, 16, 22), "track_hud": (52, 57, 70),
        "text": (236, 239, 245), "dim": (138, 145, 160),
        "ego": (255, 96, 32), "ego_stripe": (250, 250, 250),
    },
    "night": {
        "grass": (24, 38, 36), "grass_alt": (22, 35, 33), "edge_dirt": (30, 40, 38),
        "asphalt": (44, 46, 56), "track": (38, 40, 49), "shoulder": (54, 56, 66),
        "line": (214, 222, 240), "rail": (74, 64, 96), "post": (44, 40, 56),
        "ambient": (92, 98, 140), "neon": (255, 40, 170), "neon2": (0, 210, 255),
        "accent": (0, 229, 255), "accent2": (255, 46, 170), "good": (60, 240, 160),
        "warn": (255, 176, 40), "bad": (255, 64, 96), "sensor": (0, 255, 200),
        "hud_top": (18, 16, 34), "hud_bot": (8, 8, 16), "track_hud": (40, 42, 72),
        "text": (230, 236, 255), "dim": (122, 130, 168),
        "ego": (0, 214, 255), "ego_stripe": (255, 40, 170),
    },
}


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dueling_dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo', 'rule_based'])
    parser.add_argument('--weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=3, help='Number of highway lanes')
    parser.add_argument('--density', type=float, default=1.0, help='Traffic density (vehicles_density)')
    parser.add_argument('--duration', type=float, default=1000,
                        help='Episode duration in seconds (>40 enables the endless highway)')
    parser.add_argument('--no_smooth', action='store_true', help='Disable action smoothing')
    parser.add_argument('--smooth_window', type=int, default=3, help='Minimum steps between lane changes')
    parser.add_argument('--theme', type=str, default='night', choices=list(THEMES))
    parser.add_argument('--sensors', action='store_true', help='Start with the sensor overlay on')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--zoom', type=float, default=14.0, help='Pixels per meter')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    parser.add_argument('--record', type=str, default=None, help='Directory to save every frame as PNG')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Procedural art
# ---------------------------------------------------------------------------

def mix(c1, c2, t):
    """Linear blend between two RGB colors."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _rgb_surface(arr_hw3):
    return pygame.surfarray.make_surface(np.ascontiguousarray(arr_hw3.swapaxes(0, 1))).convert()


def _radial(w, h, power):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xs - (w - 1) / 2) / (w / 2), (ys - (h - 1) / 2) / (h / 2))
    return np.clip(1.0 - d, 0.0, 1.0) ** power


def radial_light(w, h, color, power=2.0):
    """Radial gradient on black, meant for additive blending."""
    w, h = max(int(w), 2), max(int(h), 2)
    return _rgb_surface((_radial(w, h, power)[..., None] * np.array(color, np.float32)).astype(np.uint8))


def cone_light(length, w0, w1, color):
    """Headlight beam pointing +x, starting at the left edge (additive)."""
    w, h = max(int(length), 2), max(int(w1), 2)
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = (np.arange(h, dtype=np.float32) - (h - 1) / 2)[:, None]
    half = (w0 + (w1 - w0) * xs) / 2
    inten = (1.0 - xs) ** 1.5 * np.sqrt(np.clip(1.0 - np.abs(ys) / half, 0.0, 1.0))
    return _rgb_surface((inten[..., None] * np.array(color, np.float32)).astype(np.uint8))


def soft_blob(w, h, alpha, color=(0, 0, 0), power=1.2):
    """Soft elliptical alpha blob (shadows, clouds, smoke)."""
    w, h = max(int(w), 2), max(int(h), 2)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((*color, 255))
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:] = (_radial(w, h, power) * alpha).astype(np.uint8).T
    del pa
    return surf


def soft_rect_shadow(w, h, alpha, blur):
    """Blurred rounded-rectangle shadow."""
    blur = max(int(blur), 1)
    W, H = int(w) + 2 * blur, int(h) + 2 * blur
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.rect(s, (0, 0, 0, alpha), (blur, blur, int(w), int(h)), border_radius=int(h * 0.35))
    k = max(2, blur // 2)
    small = pygame.transform.smoothscale(s, (max(1, W // k), max(1, H // k)))
    return pygame.transform.smoothscale(small, (W, H))


CAR_LAYOUTS = {
    "sedan": dict(rear=(0.15, 0.29), roof=(0.29, 0.60), wind=(0.60, 0.75)),
    "hatch": dict(rear=(0.05, 0.16), roof=(0.16, 0.56), wind=(0.56, 0.72)),
    "van": dict(rear=(0.03, 0.07), roof=(0.07, 0.74), wind=(0.74, 0.86)),
    "pickup": dict(rear=(0.42, 0.47), roof=(0.47, 0.63), wind=(0.63, 0.77)),
    "sport": dict(rear=(0.20, 0.32), roof=(0.32, 0.55), wind=(0.55, 0.72)),
}


def make_car_sprite(color, kind, length_px, width_px, stripe=None, ss=4):
    """Top-down car facing +x, drawn supersampled then downscaled for antialiasing."""
    L, W = int(length_px * ss), int(width_px * ss)
    pad = int(W * 0.18)
    surf = pygame.Surface((L + 2 * pad, W + 2 * pad), pygame.SRCALPHA)
    dark = mix(color, (0, 0, 0), 0.5)
    shade = mix(color, (0, 0, 0), 0.16)
    light = mix(color, (255, 255, 255), 0.12)
    glass, glass_hi = (24, 32, 46), (110, 134, 165)

    def R(fx0, fy0, fx1, fy1, c, rad=0.0):
        pygame.draw.rect(surf, c, (pad + int(L * fx0), pad + int(W * fy0),
                                   max(1, int(L * (fx1 - fx0))), max(1, int(W * (fy1 - fy0)))),
                         border_radius=int(W * rad))

    def P(points, c):
        pygame.draw.polygon(surf, c, [(pad + L * fx, pad + W * fy) for fx, fy in points])

    lay = CAR_LAYOUTS[kind]
    for fx in (0.14, 0.72):  # tyres peeking out
        R(fx, -0.03, fx + 0.14, 0.12, (16, 16, 18), 0.04)
        R(fx, 0.88, fx + 0.14, 1.03, (16, 16, 18), 0.04)
    R(0, 0.02, 1, 0.98, dark, 0.34)
    R(0.012, 0.06, 0.988, 0.94, color, 0.30)
    R(0.05, 0.36, 0.95, 0.64, light, 0.12)  # curvature highlight
    if kind == "pickup":
        R(0.04, 0.14, 0.40, 0.86, shade, 0.06)
        R(0.07, 0.21, 0.38, 0.79, mix(color, (0, 0, 0), 0.6), 0.04)
    ra, rb = lay["roof"]
    R(ra, 0.17, rb, 0.83, shade, 0.12)
    if stripe:
        R(0.0, 0.39, 1.0, 0.46, stripe)
        R(0.0, 0.54, 1.0, 0.61, stripe)
    a, b = lay["wind"]
    P([(a, 0.16), (b, 0.24), (b, 0.76), (a, 0.84)], glass)
    pygame.draw.line(surf, glass_hi, (pad + L * (a + 0.03), pad + W * 0.26),
                     (pad + L * (b - 0.03), pad + W * 0.5), max(1, ss))
    a, b = lay["rear"]
    P([(a, 0.24), (b, 0.17), (b, 0.83), (a, 0.76)], glass)
    R(ra + 0.01, 0.12, rb - 0.01, 0.17, glass)
    R(ra + 0.01, 0.83, rb - 0.01, 0.88, glass)
    wa = lay["wind"][0]
    R(wa, -0.07, wa + 0.05, 0.06, dark, 0.03)
    R(wa, 0.94, wa + 0.05, 1.07, dark, 0.03)
    R(0.955, 0.12, 0.995, 0.30, (255, 248, 214), 0.05)
    R(0.955, 0.70, 0.995, 0.88, (255, 248, 214), 0.05)
    R(0.005, 0.12, 0.035, 0.30, (170, 18, 28), 0.04)
    R(0.005, 0.70, 0.035, 0.88, (170, 18, 28), 0.04)
    if kind == "sport":
        R(0.04, 0.03, 0.08, 0.97, dark, 0.05)  # spoiler
    out = pygame.transform.smoothscale(surf, (surf.get_width() // ss, surf.get_height() // ss))
    return out.convert_alpha()


def make_tree(kind, radius_px, rng, ss=3):
    """Top-down tree / bush sprite."""
    r = radius_px * ss
    size = int(r * 2.2)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    if kind == "pine":
        layers = [(1.0, (26, 70, 44)), (0.78, (34, 88, 54)), (0.55, (46, 108, 64)), (0.3, (64, 130, 78))]
        n, rot = rng.randint(8, 11), rng.random() * math.pi
        for scale, col in layers:
            pts = [(c + r * scale * (1.0 if i % 2 == 0 else 0.62) * math.cos(rot + i * math.pi / n),
                    c + r * scale * (1.0 if i % 2 == 0 else 0.62) * math.sin(rot + i * math.pi / n))
                   for i in range(2 * n)]
            pygame.draw.polygon(surf, col, pts)
    else:
        if kind == "bush":
            pal = [(52, 104, 40), (66, 124, 48), (88, 146, 58), (118, 170, 78)]
        else:
            pal = [(40, 94, 42), (54, 114, 50), (72, 136, 58), (100, 160, 74)]
        pygame.draw.circle(surf, pal[0], (c, c), r * 0.92)
        for i in range(7):
            a = i * 2 * math.pi / 7 + rng.random() * 0.5
            pygame.draw.circle(surf, pal[1], (c + 0.38 * r * math.cos(a), c + 0.38 * r * math.sin(a)), r * 0.5)
        for _ in range(5):
            a = rng.random() * 2 * math.pi
            pygame.draw.circle(surf, pal[2], (c - 0.15 * r + 0.3 * r * math.cos(a),
                                              c - 0.15 * r + 0.3 * r * math.sin(a)), r * 0.33)
        for _ in range(4):
            a = rng.random() * 2 * math.pi
            pygame.draw.circle(surf, pal[3], (c - 0.3 * r + 0.2 * r * math.cos(a),
                                              c - 0.3 * r + 0.2 * r * math.sin(a)), r * 0.14)
    out = pygame.transform.smoothscale(surf, (size // ss, size // ss))
    return out.convert_alpha()


def draw_action_icon(surf, action, cx, cy, color):
    """Small vector icon for a DiscreteMetaAction."""
    if action in (0, 2):
        d = -1 if action == 0 else 1  # left lane = up on screen, right lane = down
        pts = [(cx, cy + 8 * d), (cx + 7, cy + d), (cx + 2, cy + d), (cx + 2, cy - 8 * d),
               (cx - 2, cy - 8 * d), (cx - 2, cy + d), (cx - 7, cy + d)]
        pygame.gfxdraw.filled_polygon(surf, pts, color)
        pygame.gfxdraw.aapolygon(surf, pts, color)
    elif action == 1:
        pygame.draw.rect(surf, color, (cx - 7, cy - 2, 14, 4), border_radius=2)
    else:
        d = 1 if action == 3 else -1
        for ox in (-4, 3):
            x = cx + ox * d
            pygame.draw.lines(surf, color, False, [(x - 3 * d, cy - 6), (x + 3 * d, cy), (x - 3 * d, cy + 6)], 3)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class GameRenderer:
    """Draws the simulation; never writes to it."""

    HUD_H = 200

    def __init__(self, args, uenv):
        """
        __init__.

        Args:
            args (argparse.Namespace): Parsed CLI arguments.
            uenv (AbstractEnv): Unwrapped highway-env instance (read-only).
        """
        pygame.init()
        pygame.display.set_caption("Highway RL - Game Demo")
        self.W, self.H = max(args.width, 1280), max(args.height, 640)
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.scene_h = self.H - self.HUD_H
        self.scene = pygame.Surface((self.W, self.scene_h)).convert()
        self.lightmap = pygame.Surface((self.W, self.scene_h)).convert()
        self.s = args.zoom
        self.fps = args.fps
        self.clock = pygame.time.Clock()
        self.theme_name = args.theme
        self.show_sensors = args.sensors
        self.record_dir = args.record
        if self.record_dir:
            os.makedirs(self.record_dir, exist_ok=True)
        self.n_frames = 0
        self.label = ALGO_LABELS.get(args.algorithm, args.algorithm.upper())
        self.info_line = f"{args.lanes} LANES · DENSITY {args.density:g}"
        self.n_episodes = args.episodes
        self.sim_freq = uenv.config["simulation_frequency"]
        self.duration = uenv.config["duration"]
        self.speed_zone = tuple(uenv.config.get("reward_speed_range", (20, 30)))
        self.frames_per_substep = self.fps / (self.sim_freq * max(args.speed, 1e-3))
        self.fx_rng = random.Random(7)  # visual effects only, independent of the simulation RNG

        self._fonts, self._text_cache, self._sprites, self._rot_cache, self._tiles = {}, {}, {}, {}, {}
        self.paused, self.request, self.skip, self.summary = False, None, False, None
        self.anim_t = 0.0
        self.shake = 0.0
        self.shake_dx = self.shake_dy = 0.0
        self.cam_x = 0.0
        self._layout(uenv)
        self._build_static_art()

    # -- setup -----------------------------------------------------------

    @property
    def T(self):
        return THEMES[self.theme_name]

    def _layout(self, uenv):
        lanes = uenv.road.network.lanes_list()
        self.lane_ys = sorted({round(float(lane.start[1]), 3) for lane in lanes})
        self.lw = float(lanes[0].width_at(0))
        self.main_top = self.lane_ys[0] - self.lw / 2
        self.main_bot = self.lane_ys[-1] + self.lw / 2
        self.shoulder = 1.5
        self.med_bot = self.main_top - self.shoulder
        self.med_top = self.med_bot - 5.0
        self.opp_bot = self.med_top - 0.8
        self.opp_top = self.opp_bot - 2 * self.lw
        self.opp_ys = [self.opp_top + self.lw * (j + 0.5) for j in range(2)]
        self.rails = [self.med_bot - 0.3, self.med_top + 0.3,
                      self.main_bot + self.shoulder + 0.3, self.opp_top - 1.1]
        self.center_y = ((self.opp_top - 4.6) + (self.main_bot + 4.5)) / 2
        self.view_m = self.W / self.s
        self.view_top = self.center_y - self.scene_h / 2 / self.s
        self.view_bot = self.center_y + self.scene_h / 2 / self.s
        self.car_l, self.car_w = float(uenv.vehicle.LENGTH), float(uenv.vehicle.WIDTH)

    def _build_static_art(self):
        s = self.s
        rng = random.Random(11)
        self.trees = []  # (kind, radius_m, sprite, shadow)
        for kind, radii in (("round", (2.2, 2.6, 3.0, 3.4)), ("pine", (1.8, 2.2, 2.6)), ("bush", (0.9, 1.1, 1.3))):
            for r in radii:
                spr = make_tree(kind, r * s, rng)
                d = spr.get_width()
                self.trees.append((kind, r, spr, soft_blob(d * 1.05, d * 0.9, 95)))
        self.big_trees = [t for t in self.trees if t[0] != "bush"] + [t for t in self.trees if t[0] == "bush"][:1]
        self.bushes = [t for t in self.trees if t[0] == "bush"]
        self.car_shadow = soft_rect_shadow(self.car_l * s, self.car_w * s, 120, 0.35 * s)
        self.glows = {
            "head": radial_light(1.0 * s, 1.0 * s, (255, 240, 210), 1.6),
            "tail": radial_light(0.8 * s, 0.8 * s, (190, 30, 30), 1.6),
            "brake": radial_light(1.9 * s, 1.9 * s, (255, 40, 30), 1.7),
            "signal": radial_light(1.5 * s, 1.5 * s, (255, 170, 30), 1.6),
            "bulb": radial_light(2.0 * s, 2.0 * s, (255, 214, 160), 1.6),
            "spark": radial_light(0.7 * s, 0.7 * s, (255, 200, 90), 1.4),
            "pool_main": radial_light(18 * s, 12 * s, (150, 128, 96), 1.3),
            "pool_opp": radial_light(14 * s, 8 * s, (140, 120, 90), 1.3),
            "under": radial_light(8 * s, 4.6 * s, (30, 150, 170), 1.4),
            "brake_pool": radial_light(5 * s, 3.4 * s, (120, 20, 20), 1.5),
        }
        self.cone = cone_light(22 * s, 1.4 * s, 7.5 * s, (165, 155, 125))
        self.cone_back = pygame.transform.rotate(self.cone, 180)
        self.cone_m = 22.0
        self.brake_day = soft_blob(1.3 * s, 1.3 * s, 170, (255, 40, 30))
        self.signal_day = soft_blob(1.2 * s, 1.2 * s, 210, (255, 170, 30))
        self.cloud = soft_blob(70 * s, 40 * s, 42, power=1.6)
        self.smoke = [soft_blob(r * s * 2, r * s * 2, 150, (70, 70, 76), 1.3) for r in (0.6, 0.9, 1.3, 1.8, 2.4, 3.0)]
        w, h = self.W, self.scene_h
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2))
        self.vignette = pygame.Surface((w, h), pygame.SRCALPHA)
        pa = pygame.surfarray.pixels_alpha(self.vignette)
        pa[:] = (np.clip((d - 0.75) / 0.7, 0, 1) ** 1.6 * 150).astype(np.uint8).T
        del pa

    def _tile(self):
        """Pre-rendered, horizontally repeating strip of ground, roads, lines and rails."""
        if self.theme_name in self._tiles:
            return self._tiles[self.theme_name]
        T, s, h = self.T, self.s, self.scene_h
        tile_m = 48.0
        tw = int(round(tile_m * s))
        tile = pygame.Surface((tw, h)).convert()
        tile.fill(T["grass"])

        def Y(y):
            return int(round(h / 2 + (y - self.center_y) * s))

        def band(y0, y1, color):
            pygame.draw.rect(tile, color, (0, Y(y0), tw, max(1, Y(y1) - Y(y0))))

        def X(m):
            return int(round(m * s))

        for k in range(8):  # mowed grass stripes
            pygame.draw.rect(tile, T["grass_alt"], (X(k * 6), 0, X(k * 6 + 3) - X(k * 6), h))
        for y0, y1 in ((self.main_top - self.shoulder, self.main_bot + self.shoulder),
                       (self.opp_top - 0.8, self.opp_bot + 0.8)):
            band(y0 - 0.35, y1 + 0.35, T["edge_dirt"])
            band(y0, y1, T["shoulder"])
        band(self.main_top, self.main_bot, T["asphalt"])
        band(self.opp_top, self.opp_bot, T["asphalt"])
        for c in self.lane_ys + self.opp_ys:  # tyre wear tracks
            for off in (-0.95, 0.95):
                band(c + off - 0.28, c + off + 0.28, T["track"])

        nrng = np.random.default_rng(3)
        arr = pygame.surfarray.pixels3d(tile)
        noise = nrng.integers(-7, 8, size=(tw, h, 1), dtype=np.int16)
        noise += (nrng.random((tw, h, 1)) < 0.012) * 16
        arr[...] = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        del arr

        lw = max(2, int(round(0.15 * s)))

        def solid(y):
            pygame.draw.rect(tile, T["line"], (0, Y(y) - lw // 2, tw, lw))

        def dashed(y):
            for k in range(4):
                pygame.draw.rect(tile, T["line"], (X(k * 12), Y(y) - lw // 2, X(4), lw))

        for ys_, top, bot in ((self.lane_ys, self.main_top, self.main_bot), (self.opp_ys, self.opp_top, self.opp_bot)):
            solid(top + 0.2)
            solid(bot - 0.2)
            for a, b in zip(ys_, ys_[1:]):
                dashed((a + b) / 2)

        shadow = mix(T["grass"], (0, 0, 0), 0.35)
        for y in self.rails:
            band(y + 0.12, y + 0.5, shadow)
            for k in range(12):
                pygame.draw.rect(tile, T["post"], (X(k * 4) - 2, Y(y) - 3, 5, 7))
            pygame.draw.rect(tile, T["rail"], (0, Y(y) - 2, tw, 4))
            pygame.draw.rect(tile, mix(T["rail"], (255, 255, 255), 0.4), (0, Y(y) - 2, tw, 1))
        self._tiles[self.theme_name] = tile
        return tile

    # -- small helpers ---------------------------------------------------

    def _font(self, size, bold=False, mono=False):
        key = (size, bold, mono)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("dejavusansmono" if mono else "dejavusans", size, bold=bold)
        return self._fonts[key]

    def text(self, s, size, color, bold=False, mono=False, outline=False):
        key = (s, size, color, bold, mono, outline)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 800:
                self._text_cache.clear()
            font = self._font(size, bold, mono)
            core = font.render(s, True, color)
            if outline:
                edge = font.render(s, True, (0, 0, 0))
                o = max(2, size // 16)
                surf = pygame.Surface((core.get_width() + 2 * o, core.get_height() + 2 * o), pygame.SRCALPHA)
                for dx in (-o, 0, o):
                    for dy in (-o, 0, o):
                        surf.blit(edge, (o + dx, o + dy))
                surf.blit(core, (o, o))
            else:
                surf = core
            self._text_cache[key] = surf
        return surf

    def sx(self, x):
        return (x - self.cam_x) * self.s + self.shake_dx

    def sy(self, y):
        return self.scene_h / 2 + (y - self.center_y) * self.s + self.shake_dy

    def _visible(self, x, margin=8.0):
        return self.cam_x - margin < x < self.cam_x + self.view_m + margin

    def _rot(self, key, surf, deg, smooth=True):
        q = round(deg * 2) / 2
        if abs(q) < 0.25:
            return surf
        rk = (key, q)
        out = self._rot_cache.get(rk)
        if out is None:
            if len(self._rot_cache) > 4000:
                self._rot_cache.clear()
            out = pygame.transform.rotozoom(surf, q, 1.0) if smooth else pygame.transform.rotate(surf, q)
            self._rot_cache[rk] = out
        return out

    def _blit_c(self, target, img, x, y, flags=0):
        target.blit(img, (x - img.get_width() / 2, y - img.get_height() / 2), special_flags=flags)

    def _car_sprite(self, look, ego=False, crashed=False, flip=False):
        key = ("car", look, ego, crashed, flip, self.theme_name if ego else None)
        spr = self._sprites.get(key)
        if spr is None:
            color, kind = (self.T["ego"], "sport") if ego else look
            if crashed:
                color = mix(color, (20, 20, 20), 0.55)
            spr = make_car_sprite(color, kind, self.car_l * self.s, self.car_w * self.s,
                                  stripe=self.T["ego_stripe"] if ego else None)
            if flip:
                spr = pygame.transform.rotate(spr, 180)
            self._sprites[key] = spr
        return key, spr

    def _banner(self, text, color, dur=1.4, size=46):
        self.banners.append([text, color, self.anim_t, dur, size])

    # -- episode / simulation sync ----------------------------------------

    def begin_episode(self, uenv, episode, seed):
        """
        Reset per-episode visual state.

        Args:
            uenv (AbstractEnv): Unwrapped env right after reset.
            episode (int): 1-based episode index.
            seed (int): Seed used for this episode's reset.
        """
        self.uenv, self.ego = uenv, uenv.vehicle
        self.ego_id = id(self.ego)
        self.episode, self.seed = episode, seed
        self.looks = {}
        self.curr = self._snapshot()
        self.prev = self.curr
        self.side = {}
        self.observed = set()
        self.overtakes, self.combo, self.last_overtake, self.pulse_t = 0, 0, -10.0, -10.0
        self.lane_changes, self.speed_sum = 0, 0.0
        self.start_x = self.curr[self.ego_id][0]
        self.substeps, self.frame_budget, self.sim_alpha = 0, 0.0, 1.0
        self.ret, self.q, self.raw_action, self.final_action = 0.0, None, 1, 1
        self.bar_disp = np.zeros(len(ACTION_LABELS))
        self.popups, self.banners, self.particles = [], [], []
        self.trail = deque(maxlen=max(4, int(0.3 * self.sim_freq * self.frames_per_substep)))
        self.ended, self.summary, self.shake = False, None, 0.0
        self._update_observed()
        self._detect_events()
        self._banner("GO!", self.T["good"], dur=1.0, size=56)
        t_end = self.anim_t + 0.6
        while self.anim_t < t_end and not self.request:
            self.frame(1.0)

    def set_decision(self, q, raw_action, final_action, reason=None):
        self.q, self.raw_action, self.final_action = q, int(raw_action), int(final_action)
        self.reason = reason

    def on_substep(self):
        """Called after every simulation substep; renders interpolated frames."""
        if self.request:
            return
        self.prev, self.curr = self.curr, self._snapshot()
        self.substeps += 1
        self._update_observed()
        self._detect_events()
        self.frame_budget += self.frames_per_substep
        n = int(self.frame_budget)
        self.frame_budget -= n
        for i in range(n):
            self.frame((i + 1) / n)
            if self.request:
                return

    def finish_episode(self, ret, crashed):
        """Short aftermath, then an episode summary card."""
        self.ret, self.ended = ret, True
        self.prev = self.curr
        if not crashed:
            self._banner("FINISH!", self.T["good"], dur=1.6, size=56)
        t_end = self.anim_t + (1.6 if crashed else 1.0)
        while self.anim_t < t_end and not self.request:
            self.frame(1.0)
        n = max(self.substeps, 1)
        self.summary = {
            "title": "CRASHED" if crashed else "EPISODE COMPLETE",
            "color": self.T["bad"] if crashed else self.T["good"],
            "rows": [("Return", f"{ret:.2f}"), ("Overtakes", str(self.overtakes)),
                     ("Avg speed", f"{self.speed_sum / n * 3.6:.0f} km/h"),
                     ("Distance", f"{self.curr[self.ego_id][0] - self.start_x:.0f} m"),
                     ("Lane changes", str(self.lane_changes))],
        }
        self.skip = False
        t_end = self.anim_t + 4.0
        while self.anim_t < t_end and not self.request and not self.skip:
            self.frame(1.0)
        self.summary = None

    def _snapshot(self):
        snap = {}
        for i, v in enumerate(self.uenv.road.vehicles):
            sig = 0
            tl, li = getattr(v, "target_lane_index", None), getattr(v, "lane_index", None)
            if tl is not None and li is not None and tl != li:
                sig = 1 if tl[2] > li[2] else -1
            acc = (getattr(v, "action", None) or {}).get("acceleration", 0.0)
            snap[id(v)] = (float(v.position[0]), float(v.position[1]), float(v.heading),
                           float(v.speed), float(acc), sig, bool(v.crashed))
            if id(v) not in self.looks:
                self.looks[id(v)] = (TRAFFIC_COLORS[(i * 7 + 3) % len(TRAFFIC_COLORS)], CAR_KINDS[(i * 5 + 1) % 4])
        return snap

    def _update_observed(self):
        """Vehicles in the agent's Kinematics observation (same query the env uses)."""
        obs = self.uenv.observation_type
        near = self.uenv.road.close_objects_to(
            self.ego, self.uenv.PERCEPTION_DISTANCE,
            count=getattr(obs, "vehicles_count", 5) - 1,
            see_behind=getattr(obs, "see_behind", False),
            sort=getattr(obs, "order", "sorted") == "sorted",
            vehicles_only=not getattr(obs, "include_obstacles", True))
        self.observed = {id(v) for v in near}

    def _detect_events(self):
        e = self.curr[self.ego_id]
        pe = self.prev.get(self.ego_id, e)
        self.speed_sum += e[3]
        if e[5] != 0 and pe[5] == 0:
            self.lane_changes += 1
        for vid, v in self.curr.items():
            if vid == self.ego_id:
                continue
            ahead = v[0] > e[0]
            if vid in self.side and self.side[vid] and not ahead and not v[6]:
                self.overtakes += 1
                self.combo = self.combo + 1 if self.anim_t - self.last_overtake < 2.5 else 1
                self.last_overtake = self.pulse_t = self.anim_t
                self.popups.append([v[0], v[1], "+1", self.anim_t])
                if self.combo >= 2:
                    self._banner(f"COMBO x{self.combo}", self.T["accent"], dur=1.1, size=40)
            self.side[vid] = ahead
            p = self.prev.get(vid)
            if v[6] and p is not None and not p[6]:
                self._spawn_crash(v[0], v[1], v[3])
        if e[6] and not pe[6]:
            self._spawn_crash(e[0], e[1], e[3])
            self.shake = 16.0
            self._banner("CRASH!", self.T["bad"], dur=2.0, size=64)

    def _spawn_crash(self, x, y, speed):
        r = self.fx_rng
        for _ in range(46):
            a, sp = r.uniform(0, 2 * math.pi), r.uniform(4, 16)
            life = r.uniform(0.35, 0.9)
            self.particles.append([x, y, math.cos(a) * sp + speed * 0.3, math.sin(a) * sp, life, life, "spark"])
        for _ in range(16):
            life = r.uniform(1.6, 2.8)
            self.particles.append([x + r.uniform(-1.5, 1.5), y + r.uniform(-0.8, 0.8),
                                   r.uniform(-1.5, 1.5) + speed * 0.1, r.uniform(-1.0, 1.0), life, life, "smoke"])

    # -- frame loop --------------------------------------------------------

    def _pump(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.request = "quit"
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    self.request = "quit"
                elif k in (pygame.K_SPACE, pygame.K_RETURN):
                    if self.summary:
                        self.skip = True
                    else:
                        self.paused = not self.paused
                elif k == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif k == pygame.K_t:
                    self.theme_name = "day" if self.theme_name == "night" else "night"
                elif k == pygame.K_r:
                    self.request = "restart"
                elif k == pygame.K_n:
                    self.request = "next"

    def frame(self, alpha):
        """Render one frame interpolated between the previous and current substep."""
        self.sim_alpha = alpha
        self._pump()
        while self.paused and not self.request:
            self._render(0.0)
            self._pump()
        if not self.request:
            self._render(1.0 / self.fps)

    def _render(self, dt):
        self.anim_t += dt
        self._update_particles(dt)
        st = self._interp(self.sim_alpha)
        self._draw_scene(st, dt)
        self.screen.blit(self.scene, (0, 0))
        self._draw_hud(st)
        if self.summary:
            self._draw_summary()
        if self.paused:
            self._draw_paused()
        pygame.display.flip()
        self.clock.tick(self.fps)
        if self.record_dir and dt > 0:
            self.n_frames += 1
            pygame.image.save(self.screen, os.path.join(self.record_dir, f"frame_{self.n_frames:06d}.png"))

    def _interp(self, a):
        out = {}
        for vid, c in self.curr.items():
            p = self.prev.get(vid, c)
            if abs(c[0] - p[0]) > 50.0:  # recycled by EndlessHighway: jump, don't slide
                p = c
            dh = (c[2] - p[2] + math.pi) % (2 * math.pi) - math.pi
            out[vid] = (p[0] + (c[0] - p[0]) * a, p[1] + (c[1] - p[1]) * a, p[2] + dh * a,
                        p[3] + (c[3] - p[3]) * a, c[4], c[5], c[6])
        return out

    def _update_particles(self, dt):
        if dt <= 0:
            return
        alive = []
        for p in self.particles:
            p[4] -= dt
            if p[4] <= 0:
                continue
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            damp = 0.90 if p[6] == "spark" else 0.97
            p[2] *= damp ** (dt * 60)
            p[3] *= damp ** (dt * 60)
            alive.append(p)
        self.particles = alive
        self.shake *= 0.9 ** (dt * 60)

    # -- scene -------------------------------------------------------------

    def _sim_time(self):
        return (self.substeps - 1 + self.sim_alpha) / self.sim_freq

    def _lamp_xs(self):
        sp = 40.0
        lo = int((self.cam_x - 12) // sp)
        hi = int((self.cam_x + self.view_m + 12) // sp)
        return [k * sp + 20 for k in range(lo, hi + 1)]

    def _opposite_cars(self):
        out, t = [], self._sim_time()
        for j, (y, v, sp) in enumerate(((self.opp_ys[0], 22.0, 46.0), (self.opp_ys[1], 28.0, 58.0))):
            lo = self.cam_x - 10 + v * t
            hi = self.cam_x + self.view_m + 10 + v * t
            for k in range(int(lo // sp) - 1, int(hi // sp) + 2):
                r = random.Random(k * 92821 + j * 689287)
                if r.random() < 0.35:
                    continue
                x = k * sp + r.uniform(0, sp * 0.5) - v * t
                if self._visible(x):
                    out.append((x, y, (TRAFFIC_COLORS[r.randrange(len(TRAFFIC_COLORS))], CAR_KINDS[r.randrange(4)])))
        return out

    def _scenery(self):
        items = []
        regions = ((0, self.view_top - 3.0, self.opp_top - 1.6, 6.0, self.big_trees, (0, 1, 1, 2)),
                   (1, self.main_bot + self.shoulder + 0.9, self.view_bot + 3.0, 6.0, self.big_trees, (0, 1, 1, 2)),
                   (2, self.med_top + 1.6, self.med_bot - 1.6, 5.0, self.bushes, (0, 0, 1)))
        for rid, lo, hi, cell, pool, counts in regions:
            if hi <= lo:
                continue
            for k in range(int((self.cam_x - 8) // cell), int((self.cam_x + self.view_m + 8) // cell) + 1):
                r = random.Random(k * 73856093 + rid * 19349663 + 12345)
                for _ in range(r.choice(counts)):
                    kind, rad, spr, shadow = pool[r.randrange(len(pool))]
                    a, b = (lo, hi - rad) if rid == 0 else (lo + rad, hi) if rid == 1 else (lo, hi)
                    if b <= a:
                        continue
                    items.append((r.uniform(a, b), k * cell + r.random() * cell, rad, spr, shadow))
        items.sort(key=lambda it: it[0])
        return items

    def _draw_scene(self, st, dt):
        T, s, sc = self.T, self.s, self.scene
        night = T["ambient"] is not None
        e = st[self.ego_id]
        self.cam_x = e[0] - 0.26 * self.view_m
        if self.shake > 0.3:
            self.shake_dx = self.fx_rng.uniform(-self.shake, self.shake)
            self.shake_dy = self.fx_rng.uniform(-self.shake, self.shake) * 0.6
        else:
            self.shake_dx = self.shake_dy = 0.0

        sc.fill(T["grass"])
        tile = self._tile()
        tw = tile.get_width()
        x = -((self.cam_x * s) % tw) + self.shake_dx
        while x < self.W:
            sc.blit(tile, (x, self.shake_dy))
            x += tw

        for y, x, rad, spr, shadow in self._scenery():
            px, py = self.sx(x), self.sy(y)
            self._blit_c(sc, shadow, px + 0.35 * rad * s, py + 0.45 * rad * s)
            self._blit_c(sc, spr, px, py)

        lamps = self._lamp_xs()
        med_y = (self.med_top + self.med_bot) / 2
        arm_a, arm_b = self.main_top + 1.0, self.opp_bot - 1.0

        opp = self._opposite_cars()
        for x, y, look in opp:
            self._blit_c(sc, self.car_shadow, self.sx(x) + 0.25 * s, self.sy(y) + 0.35 * s)
            self._blit_c(sc, self._car_sprite(look, flip=True)[1], self.sx(x), self.sy(y))

        vis = [(vid, v) for vid, v in st.items() if self._visible(v[0])]
        vis.sort(key=lambda it: it[0] == self.ego_id)
        for vid, v in vis:
            deg = -math.degrees(v[2])
            ego = vid == self.ego_id
            px, py = self.sx(v[0]), self.sy(v[1])
            sh = self._rot("shadow", self.car_shadow, deg)
            self._blit_c(sc, sh, px + 0.25 * s, py + 0.35 * s)
            key, spr = self._car_sprite(None if ego else self.looks.get(vid, (TRAFFIC_COLORS[0], "sedan")),
                                        ego=ego, crashed=v[6])
            self._blit_c(sc, self._rot(key, spr, deg), px, py)

        post, head = T["post"], mix(T["rail"], (255, 255, 255), 0.5)
        for lx in lamps:
            px = self.sx(lx)
            pygame.draw.line(sc, post, (px, self.sy(arm_b)), (px, self.sy(arm_a)), max(2, int(0.18 * s)))
            pygame.draw.circle(sc, post, (px, self.sy(med_y)), max(3, int(0.35 * s)))
            for ay in (arm_a, arm_b):
                pygame.draw.rect(sc, head, (px - 0.45 * s, self.sy(ay) - 0.25 * s, 0.9 * s, 0.5 * s), border_radius=3)

        if night:
            self._draw_night_lights(vis, opp, lamps, arm_a, arm_b)
        else:
            self._draw_day_lights(vis)
            t = self._sim_time()
            for k in range(int((self.cam_x - 40 - 4 * t) // 160) - 1, int((self.cam_x + self.view_m + 40 - 4 * t) // 160) + 1):
                r = random.Random(k * 7919 + 11)
                cx = k * 160 + r.uniform(0, 80) + 4.0 * t
                if self._visible(cx, 40):
                    self._blit_c(sc, self.cloud, self.sx(cx), self.sy(self.center_y + r.uniform(-12, 12)))

        if self.show_sensors:
            self._draw_sensors(st)
        self._draw_effects(night)
        sc.blit(self.vignette, (0, 0))
        self._draw_scene_text()

    def _car_points(self, v):
        c, s_ = math.cos(v[2]), math.sin(v[2])
        hl, hw = self.car_l / 2, self.car_w * 0.32
        return (v[0] + c * hl, v[1] + s_ * hl), (v[0] - c * hl, v[1] - s_ * hl), (-s_ * hw, c * hw), (c, s_)

    def _draw_night_lights(self, vis, opp, lamps, arm_a, arm_b):
        T, G, s, sc, lm = self.T, self.glows, self.s, self.scene, self.lightmap
        ADD = pygame.BLEND_ADD
        lm.fill(T["ambient"])
        for lx in lamps:
            self._blit_c(lm, G["pool_main"], self.sx(lx), self.sy(arm_a + 3.0), ADD)
            self._blit_c(lm, G["pool_opp"], self.sx(lx), self.sy(arm_b - 2.0), ADD)
        for vid, v in vis:
            front, rear, n, d = self._car_points(v)
            if not v[6]:
                cone = self._rot("cone", self.cone, -math.degrees(v[2]), smooth=False)
                self._blit_c(lm, cone, self.sx(front[0] + d[0] * self.cone_m / 2), self.sy(front[1] + d[1] * self.cone_m / 2), ADD)
            if v[4] < -0.5:
                self._blit_c(lm, G["brake_pool"], self.sx(rear[0]), self.sy(rear[1]), ADD)
            if vid == self.ego_id:
                self._blit_c(lm, G["under"], self.sx(v[0]), self.sy(v[1]), ADD)
        for x, y, _ in opp:
            self._blit_c(lm, self.cone_back, self.sx(x - self.car_l / 2 - self.cone_m / 2), self.sy(y), ADD)
        sc.blit(lm, (0, 0), special_flags=pygame.BLEND_MULT)

        for y, col in ((self.rails[0], T["neon"]), (self.rails[2], T["neon"]), (self.rails[1], T["neon2"]), (self.rails[3], T["neon2"])):
            py = self.sy(y)
            halo = self._sprites.get(("halo", col))
            if halo is None:
                hh = int(1.4 * s)
                ramp = (1 - np.abs(np.linspace(-1, 1, hh, dtype=np.float32))) ** 2
                arr = np.repeat((ramp[:, None, None] * np.array(col, np.float32) * 0.55)[:, :1], self.W + 64, axis=1)
                halo = self._sprites[("halo", col)] = _rgb_surface(arr.astype(np.uint8))
            sc.blit(halo, (self.shake_dx - 32, py - halo.get_height() / 2), special_flags=ADD)
            pygame.draw.line(sc, mix(col, (255, 255, 255), 0.35), (0, py), (self.W, py), 2)
        for lx in lamps:
            for ay in (arm_a, arm_b):
                self._blit_c(sc, G["bulb"], self.sx(lx), self.sy(ay), ADD)

        blink = (self.anim_t * 2.2) % 1.0 < 0.55
        for vid, v in vis:
            front, rear, n, d = self._car_points(v)
            for side in (-1, 1):
                fx, fy = front[0] + n[0] * side, front[1] + n[1] * side
                rx, ry = rear[0] + n[0] * side, rear[1] + n[1] * side
                if not v[6]:
                    self._blit_c(sc, G["head"], self.sx(fx), self.sy(fy), ADD)
                self._blit_c(sc, G["brake"] if v[4] < -0.5 else G["tail"], self.sx(rx), self.sy(ry), ADD)
                if blink and v[5] == side:
                    self._blit_c(sc, G["signal"], self.sx(fx), self.sy(fy), ADD)
                    self._blit_c(sc, G["signal"], self.sx(rx), self.sy(ry), ADD)
        for x, y, _ in opp:
            for side in (-1, 1):
                self._blit_c(sc, G["head"], self.sx(x - self.car_l / 2), self.sy(y + side * self.car_w * 0.32), ADD)
                self._blit_c(sc, G["tail"], self.sx(x + self.car_l / 2), self.sy(y + side * self.car_w * 0.32), ADD)

        e = self.curr[self.ego_id]
        front, rear, n, d = self._car_points(self._interp(self.sim_alpha)[self.ego_id])
        self.trail.append((rear[0] - n[0], rear[1] - n[1], rear[0] + n[0], rear[1] + n[1]))
        if not e[6] and len(self.trail) > 1:
            base = mix(T["asphalt"], (0, 0, 0), 0.6)
            pts = list(self.trail)
            for i in range(1, len(pts)):
                col = mix(base, T["accent2"], i / len(pts))
                for o in (0, 2):
                    pygame.draw.line(sc, col, (self.sx(pts[i - 1][o]), self.sy(pts[i - 1][o + 1])),
                                     (self.sx(pts[i][o]), self.sy(pts[i][o + 1])), 3)

    def _draw_day_lights(self, vis):
        blink = (self.anim_t * 2.2) % 1.0 < 0.55
        for vid, v in vis:
            front, rear, n, d = self._car_points(v)
            for side in (-1, 1):
                rx, ry = rear[0] + n[0] * side, rear[1] + n[1] * side
                if v[4] < -0.5:
                    self._blit_c(self.scene, self.brake_day, self.sx(rx), self.sy(ry))
                if blink and v[5] == side:
                    fx, fy = front[0] + n[0] * side, front[1] + n[1] * side
                    self._blit_c(self.scene, self.signal_day, self.sx(fx), self.sy(fy))
                    self._blit_c(self.scene, self.signal_day, self.sx(rx), self.sy(ry))

    def _draw_sensors(self, st):
        T, s, sc = self.T, self.s, self.scene
        e = st[self.ego_id]
        ox, oy = self.sx(e[0] + self.car_l / 2), self.sy(e[1])
        for vid in self.observed:
            v = st.get(vid)
            if v is None:
                continue
            dx = v[0] - e[0]
            gap = abs(dx) - self.car_l
            closing = (e[3] - v[3]) if dx > 0 else (v[3] - e[3])
            ttc = gap / closing if closing > 0.1 and gap > 0 else math.inf
            col = T["bad"] if ttc < 2.5 else T["warn"] if ttc < 5 else T["sensor"]
            tx, ty = self.sx(v[0]), self.sy(v[1])
            dist = math.hypot(tx - ox, ty - oy)
            if dist > 1:
                n = int(dist // 10)
                for i in range(0, n, 2):
                    a, b = i / max(n, 1), min((i + 1) / max(n, 1), 1)
                    pygame.draw.aaline(sc, col, (ox + (tx - ox) * a, oy + (ty - oy) * a), (ox + (tx - ox) * b, oy + (ty - oy) * b))
            hw, hh, c = self.car_l * s / 2 + 5, self.car_w * s / 2 + 5, 8
            for sxn in (-1, 1):
                for syn in (-1, 1):
                    cx, cy = tx + sxn * hw, ty + syn * hh
                    pygame.draw.line(sc, col, (cx, cy), (cx - sxn * c, cy), 2)
                    pygame.draw.line(sc, col, (cx, cy), (cx, cy - syn * c), 2)
            label = f"{math.hypot(dx, v[1] - e[1]):.0f} m" + (f" · TTC {ttc:.1f}s" if ttc < 10 else "")
            img = self.text(label, 12, col, bold=True, outline=True)
            sc.blit(img, (tx - img.get_width() / 2, ty - hh - img.get_height() - 2))

    def _draw_effects(self, night):
        T, s, sc = self.T, self.s, self.scene
        for p in self.particles:
            x, y, frac = self.sx(p[0]), self.sy(p[1]), p[4] / p[5]
            if p[6] == "spark":
                if night:
                    self._blit_c(sc, self.glows["spark"], x, y, pygame.BLEND_ADD)
                else:
                    pygame.draw.circle(sc, mix((255, 120, 20), (255, 230, 120), frac), (x, y), max(1, 3 * frac))
            else:
                img = self.smoke[min(len(self.smoke) - 1, int((1 - frac) * len(self.smoke)))]
                img.set_alpha(int(220 * frac))
                self._blit_c(sc, img, x, y)
        for p in self.popups[:]:
            age = self.anim_t - p[3]
            if age > 1.1:
                self.popups.remove(p)
                continue
            img = self.text(p[2], 26, T["accent"], bold=True, outline=True)
            img.set_alpha(int(255 * min(1, (1.1 - age) / 0.4)))
            self._blit_c(sc, img, self.sx(p[0]), self.sy(p[1]) - 26 - 40 * age)
            img.set_alpha(255)

    def _draw_scene_text(self):
        T, sc = self.T, self.scene
        dot = self.text("●", 14, T["accent"])
        title = self.text(self.label, 15, T["text"], bold=True)
        info = self.text(f"EPISODE {self.episode}/{self.n_episodes} · SEED {self.seed} · {self.info_line}", 13, T["dim"], bold=True)
        w = dot.get_width() + title.get_width() + info.get_width() + 48
        pill = pygame.Surface((w, 34), pygame.SRCALPHA)
        pygame.draw.rect(pill, (0, 0, 0, 150), pill.get_rect(), border_radius=17)
        pill.blit(dot, (14, 8))
        pill.blit(title, (20 + dot.get_width(), 7))
        pill.blit(info, (34 + dot.get_width() + title.get_width(), 9))
        sc.blit(pill, (18, 16))

        hint = self.text("SPACE pause · S sensors · T theme · R restart · N next · ESC quit", 12, (220, 224, 235))
        pill = pygame.Surface((hint.get_width() + 28, 28), pygame.SRCALPHA)
        pygame.draw.rect(pill, (0, 0, 0, 120), pill.get_rect(), border_radius=14)
        pill.blit(hint, (14, 6))
        sc.blit(pill, (self.W - pill.get_width() - 18, 19))

        row = 0
        for b in self.banners[:]:
            text, col, t0, dur, size = b
            age = self.anim_t - t0
            if age > dur:
                self.banners.remove(b)
                continue
            img = self.text(text, size, col, bold=True, outline=True)
            pop = 1 + 0.6 * max(0.0, 1 - age / 0.18)
            if pop > 1.01:
                img = pygame.transform.smoothscale(img, (int(img.get_width() * pop), int(img.get_height() * pop)))
            img.set_alpha(int(255 * min(1.0, (dur - age) / 0.4)))
            self._blit_c(sc, img, self.W / 2, 110 + row * (size + 16))
            row += 1

    # -- HUD ---------------------------------------------------------------

    def _hud_bg(self):
        key = ("hud", self.theme_name)
        bg = self._sprites.get(key)
        if bg is None:
            T = self.T
            ramp = np.linspace(0, 1, self.HUD_H, dtype=np.float32)[:, None, None]
            arr = np.array(T["hud_top"], np.float32) * (1 - ramp) + np.array(T["hud_bot"], np.float32) * ramp
            bg = _rgb_surface(np.repeat(arr, self.W, axis=1).astype(np.uint8))
            for x in range(self.W):
                f = 1 - abs(x / self.W - 0.5) * 2
                bg.set_at((x, 0), mix(T["hud_top"], T["accent"], 0.25 + 0.75 * f))
                bg.set_at((x, 1), mix(T["hud_top"], T["accent"], 0.1 + 0.3 * f))
            for sx_ in (248, 748, 1074):
                pygame.draw.line(bg, mix(T["hud_top"], T["dim"], 0.25), (sx_, 22), (sx_, self.HUD_H - 22))
            self._sprites[key] = bg
        return bg

    def _ring(self, surf, color, cx, cy, r_out, r_in, a0, a1):
        if abs(a1 - a0) < 1e-3:
            return
        n = max(2, int(abs(a1 - a0) / 0.03))
        angs = np.linspace(a0, a1, n)
        pts = [(cx + r_out * math.cos(a), cy - r_out * math.sin(a)) for a in angs]
        pts += [(cx + r_in * math.cos(a), cy - r_in * math.sin(a)) for a in angs[::-1]]
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        pygame.gfxdraw.filled_polygon(surf, pts, color)
        pygame.gfxdraw.aapolygon(surf, pts, color)

    def _gauge(self, surf, cx, cy, r, kmh):
        T, vmax = self.T, 160.0

        def ang(v):
            return math.radians(225 - 270 * min(max(v, 0), vmax) / vmax)

        self._ring(surf, T["track_hud"], cx, cy, r, r - 10, ang(0), ang(vmax))
        z0, z1 = self.speed_zone[0] * 3.6, self.speed_zone[1] * 3.6
        self._ring(surf, mix(T["good"], T["hud_bot"], 0.35), cx, cy, r + 6, r + 3, ang(z0), ang(z1))
        self._ring(surf, T["accent"], cx, cy, r, r - 10, ang(0), ang(kmh))
        for v in range(0, int(vmax) + 1, 20):
            a, major = ang(v), v % 40 == 0
            r1, r2 = r - 15, r - (24 if major else 19)
            pygame.draw.aaline(surf, T["text"] if major else T["dim"], (cx + r1 * math.cos(a), cy - r1 * math.sin(a)),
                               (cx + r2 * math.cos(a), cy - r2 * math.sin(a)))
            if major:
                img = self.text(str(v), 11, T["dim"], bold=True)
                self._blit_c(surf, img, cx + (r - 36) * math.cos(a), cy - (r - 36) * math.sin(a))
        a = ang(kmh)
        tip = (cx + (r - 14) * math.cos(a), cy - (r - 14) * math.sin(a))
        nrm = (math.cos(a + math.pi / 2) * 4, -math.sin(a + math.pi / 2) * 4)
        pts = [(int(tip[0]), int(tip[1])), (int(cx + nrm[0]), int(cy + nrm[1])), (int(cx - nrm[0]), int(cy - nrm[1]))]
        pygame.gfxdraw.filled_polygon(surf, pts, T["accent2"])
        pygame.gfxdraw.aapolygon(surf, pts, T["accent2"])
        pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), 7, T["text"])
        pygame.gfxdraw.aacircle(surf, int(cx), int(cy), 7, T["text"])
        self._blit_c(surf, self.text(f"{kmh:.0f}", 30, T["text"], bold=True, mono=True), cx, cy + r * 0.74)
        self._blit_c(surf, self.text("km/h", 12, T["dim"], bold=True), cx, cy + r * 0.74 + 22)

    def _decision(self, surf, r):
        T = self.T
        surf.blit(self.text("AI DECISION", 13, T["dim"], bold=True), (r.x, r.y))
        subtitle = "Q-VALUES" if self.q is not None else "RULE FIRED"
        surf.blit(self.text(subtitle, 11, mix(T["dim"], T["hud_bot"], 0.3), bold=True), (r.x + 108, r.y + 2))
        if self.raw_action != self.final_action:
            tag = self.text(f"SMOOTHER  {ACTION_LABELS[self.raw_action]} → {ACTION_LABELS[self.final_action]}", 11, T["warn"], bold=True)
            surf.blit(tag, (r.right - tag.get_width(), r.y + 2))
        q = self.q
        if q is not None:
            lo, hi = float(q.min()), float(q.max())
            norm = (q - lo) / (hi - lo) if hi - lo > 1e-6 else np.ones_like(q)
            self.bar_disp += (norm - self.bar_disp) * 0.25
        bar_x, bar_w, row_h = r.x + 140, r.w - 140 - 74, 28
        for i, name in enumerate(ACTION_LABELS):
            y = r.y + 28 + i * row_h
            chosen = i == self.final_action
            col = T["accent"] if chosen else T["dim"]
            if chosen:
                pygame.draw.rect(surf, mix(T["hud_top"], T["accent"], 0.13), (r.x - 6, y - 2, r.w + 8, row_h - 4), border_radius=6)
            draw_action_icon(surf, i, r.x + 12, y + 10, col)
            surf.blit(self.text(name, 13, T["text"] if chosen else T["dim"], bold=chosen), (r.x + 32, y + 2))
            if q is not None:
                pygame.draw.rect(surf, T["track_hud"], (bar_x, y + 5, bar_w, 10), border_radius=5)
                fw = int(bar_w * (0.05 + 0.95 * float(self.bar_disp[i])))
                pygame.draw.rect(surf, col if chosen else mix(T["dim"], T["track_hud"], 0.35), (bar_x, y + 5, fw, 10), border_radius=5)
                surf.blit(self.text(f"{q[i]:7.2f}", 13, T["text"] if chosen else T["dim"], mono=True), (bar_x + bar_w + 8, y + 1))
        if q is None:
            reason = getattr(self, "reason", None) or "no Q-values for this agent"
            surf.blit(self.text(reason, 12, T["accent"], bold=True), (bar_x, r.y + 28 + self.final_action * row_h + 2))

    def _stats(self, surf, r, st):
        T = self.T
        e = st[self.ego_id]
        surf.blit(self.text("OVERTAKES", 13, T["dim"], bold=True), (r.x, r.y))
        img = self.text(str(self.overtakes), 56, T["accent"], bold=True, mono=True)
        pop = 1 + 0.4 * math.exp(-(self.anim_t - self.pulse_t) * 7)
        if pop > 1.01:
            img = pygame.transform.smoothscale(img, (int(img.get_width() * pop), int(img.get_height() * pop)))
        surf.blit(img, (r.x + 40 - img.get_width() / 2 + 20, r.y + 52 - img.get_height() / 2))
        if self.combo >= 2 and self.anim_t - self.last_overtake < 2.5:
            surf.blit(self.text(f"x{self.combo} COMBO", 12, T["accent2"], bold=True), (r.x, r.y + 88))

        cx = r.x + 150
        surf.blit(self.text("RETURN", 13, T["dim"], bold=True), (cx, r.y))
        surf.blit(self.text(f"{self.ret:.2f}", 24, T["text"], bold=True, mono=True), (cx, r.y + 18))
        surf.blit(self.text("LANE", 13, T["dim"], bold=True), (cx, r.y + 56))
        lane = int(np.argmin([abs(e[1] - c) for c in self.lane_ys]))
        for i in range(len(self.lane_ys)):
            col = T["accent"] if i == lane else T["track_hud"]
            pygame.draw.rect(surf, col, (cx + i * 24, r.y + 78, 18, 8), border_radius=4)

        t = min((self.substeps - 1 + self.sim_alpha) / self.sim_freq, self.duration) if self.substeps else 0.0
        surf.blit(self.text("TIME", 13, T["dim"], bold=True), (r.x, r.y + 118))
        tt = self.text(f"{t:4.1f} / {self.duration:.0f} s", 13, T["text"], bold=True, mono=True)
        surf.blit(tt, (r.right - tt.get_width(), r.y + 118))
        pygame.draw.rect(surf, T["track_hud"], (r.x, r.y + 140, r.w, 8), border_radius=4)
        pygame.draw.rect(surf, T["good"] if not e[6] else T["bad"], (r.x, r.y + 140, int(r.w * t / self.duration), 8), border_radius=4)

    def _minimap(self, surf, r, st):
        T = self.T
        rng_m = 250
        surf.blit(self.text("RADAR", 13, T["dim"], bold=True), (r.x, r.y))
        surf.blit(self.text(f"±{rng_m} m", 11, mix(T["dim"], T["hud_bot"], 0.3), bold=True), (r.x + 62, r.y + 2))
        box = pygame.Rect(r.x, r.y + 24, r.w, r.h - 24)
        pygame.draw.rect(surf, mix(T["hud_bot"], (0, 0, 0), 0.4), box, border_radius=8)
        pygame.draw.rect(surf, T["track_hud"], box, 1, border_radius=8)
        e = st[self.ego_id]
        n = len(self.lane_ys)
        inner = box.inflate(-12, -16)

        def mx(x):
            return inner.centerx + (x - e[0]) / rng_m * inner.w / 2

        def my(y):
            return inner.y + (y - self.main_top) / (self.main_bot - self.main_top) * inner.h

        for i in range(1, n):
            yy = my(self.main_top + i * self.lw)
            for xx in range(inner.x, inner.right, 10):
                pygame.draw.line(surf, T["track_hud"], (xx, yy), (min(xx + 5, inner.right), yy))
        x0, x1 = max(mx(self.cam_x), inner.x), min(mx(self.cam_x + self.view_m), inner.right)
        pygame.draw.rect(surf, mix(T["hud_bot"], T["accent"], 0.35), (x0, inner.y - 4, x1 - x0, inner.h + 8), 1, border_radius=4)
        mh = max(4, int(inner.h / n * 0.45))
        for vid, v in st.items():
            px = mx(v[0])
            if not inner.x <= px <= inner.right or vid == self.ego_id:
                continue
            col = T["bad"] if v[6] else T["accent2"] if vid in self.observed else T["dim"]
            pygame.draw.rect(surf, col, (px - 3, my(v[1]) - mh / 2, 7, mh), border_radius=2)
        pygame.draw.rect(surf, T["accent"], (mx(e[0]) - 5, my(e[1]) - mh / 2 - 1, 10, mh + 2), border_radius=3)

    def _draw_hud(self, st):
        y0 = self.scene_h
        self.screen.blit(self._hud_bg(), (0, y0))
        self._gauge(self.screen, 130, y0 + 98, 78, st[self.ego_id][3] * 3.6)
        self._decision(self.screen, pygame.Rect(270, y0 + 18, 456, 170))
        self._stats(self.screen, pygame.Rect(772, y0 + 18, 280, 170), st)
        self._minimap(self.screen, pygame.Rect(1096, y0 + 18, self.W - 1096 - 22, 162), st)

    def _draw_summary(self):
        T, sm = self.T, self.summary
        w, h = 460, 300
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(card, (*T["hud_bot"], 235), card.get_rect(), border_radius=18)
        pygame.draw.rect(card, sm["color"], card.get_rect(), 2, border_radius=18)
        self._blit_c(card, self.text(sm["title"], 30, sm["color"], bold=True), w / 2, 42)
        for i, (k, v) in enumerate(sm["rows"]):
            y = 86 + i * 34
            card.blit(self.text(k, 16, T["dim"], bold=True), (44, y))
            val = self.text(v, 18, T["text"], bold=True, mono=True)
            card.blit(val, (w - 44 - val.get_width(), y - 1))
        self._blit_c(card, self.text("SPACE continue · R replay · ESC quit", 12, T["dim"]), w / 2, h - 24)
        self.screen.blit(card, ((self.W - w) / 2, (self.scene_h - h) / 2 + 20))

    def _draw_paused(self):
        shade = pygame.Surface((self.W, self.scene_h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 110))
        self.screen.blit(shade, (0, 0))
        self._blit_c(self.screen, self.text("PAUSED", 64, self.T["text"], bold=True, outline=True), self.W / 2, self.scene_h / 2)


def q_values(agent, state):
    """Read-only forward pass for the HUD; returns None for agents without a Q-network."""
    net = getattr(agent, "q_network", None)
    if net is None:
        return None
    with torch.no_grad():
        q = net(torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0))
    return q.squeeze(0).cpu().numpy()


def run(args):
    """Run episodes exactly like evaluate.py, rendering them with GameRenderer."""
    set_seed(args.seed)
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': args.lanes,
        'vehicles_density': args.density,
        'duration': args.duration,
        'ego_spacing': 1.5,
    }
    env = make_long_episode(gymnasium.make("highway-v0", config=env_config), args.duration)
    uenv = env.unwrapped
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    if args.algorithm == 'rule_based':
        from baselines.rule_based import RuleBasedPolicy
        agent, smoother = RuleBasedPolicy(lanes_count=args.lanes), None
    else:
        agent = load_agent(args.algorithm, state_dim, env.action_space.n, args.weights)
        smoother = None if args.no_smooth else ActionSmoother(cooldown=args.smooth_window)

    renderer = None
    episode = 1
    while episode <= args.episodes:
        seed = args.seed + episode - 1
        state, _ = env.reset(seed=seed)
        state = state.reshape(-1)
        if renderer is None:
            renderer = GameRenderer(args, uenv)
            # highway-env calls this after each intermediate substep; it only triggers drawing
            uenv._automatic_rendering = renderer.on_substep
        renderer.begin_episode(uenv, episode, seed)
        if smoother:
            smoother.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        done = truncated = False
        episode_return, info = 0.0, {}
        while not (done or truncated) and not renderer.request:
            raw = agent.select_action(state, evaluate=True)
            action = smoother.filter(raw, state) if smoother else raw
            renderer.set_decision(q_values(agent, state), raw, action, getattr(agent, "reason", None))
            state, reward, done, truncated, info = env.step(action)
            state = state.reshape(-1)
            episode_return += reward
            renderer.ret = episode_return
            renderer.on_substep()  # last substep of the step is not auto-rendered by the env

        if not renderer.request:
            crashed = bool(info.get("crashed", done and not truncated))
            print(f"Episode {episode} (seed {seed}): return={episode_return:.2f} "
                  f"overtakes={renderer.overtakes} crashed={crashed}")
            renderer.finish_episode(episode_return, crashed)
        request, renderer.request = renderer.request, None
        if request == "quit":
            break
        if request != "restart":
            episode += 1

    env.close()
    pygame.quit()


if __name__ == "__main__":
    run(parse_args())
