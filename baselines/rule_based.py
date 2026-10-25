"""
Simple rule-based driver for highway-v0.

Uses only the agent's Kinematics observation (ego + 4 nearest vehicles), converted
back to meters and m/s, and applies a handful of readable rules:

1. Danger (car close ahead or TTC short): change lane if a neighbour lane is free,
   otherwise brake.
2. Slower car ahead: overtake on the left if that lane is free, otherwise keep speed.
3. Road clear: return to the right lane (reward bonus) when it is free.
4. Otherwise speed up to the top target speed.

It only sees 4 vehicles, so it can miss a car in the target lane and crash; that is
expected for a baseline.

Usage:
    python baselines/rule_based.py
    python baselines/rule_based.py --episodes 100 --no_render
    python baselines/rule_based.py --lanes 4 --density 2 --seed 6
"""

import argparse
import sys
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401  (registers highway-v0)
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.env.endless_highway import make_long_episode  # noqa: E402


# Kinematics normalisation ranges used by highway-env (Vehicle.MAX_SPEED = 40, lane width = 4 m)
X_RANGE = 5.0 * 40.0
VX_RANGE = 2.0 * 40.0
LANE_WIDTH = 4.0
CAR_LENGTH = 5.0


class RuleBasedPolicy:
    """
    Rule-based policy over the normalised Kinematics observation.

    Actions (DiscreteMetaAction): 0 lane left, 1 idle, 2 lane right, 3 faster, 4 slower.
    """

    LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4

    def __init__(self, lanes_count: int = 3, danger_gap: float = 12.0, danger_ttc: float = 3.0,
                 follow_gap: float = 30.0, follow_ttc: float = 6.0, top_speed: float = 29.0,
                 lane_change_cooldown: int = 2):
        """
        __init__.

        Args:
            lanes_count (int): Number of lanes in the environment.
            danger_gap (float): Bumper gap in meters that triggers an evasive action.
            danger_ttc (float): Time-to-collision in seconds that triggers an evasive action.
            follow_gap (float): Gap in meters under which a slower car ahead is worth overtaking.
            follow_ttc (float): TTC in seconds under which a slower car ahead is worth overtaking.
            top_speed (float): Speed in m/s above which the policy stops accelerating.
            lane_change_cooldown (int): Steps to wait after a lane change before a non-urgent one.
        """
        self.lanes_count = lanes_count
        self.danger_gap = danger_gap
        self.danger_ttc = danger_ttc
        self.follow_gap = follow_gap
        self.follow_ttc = follow_ttc
        self.top_speed = top_speed
        self.lane_change_cooldown = lane_change_cooldown
        self.steps_since_lane_change = lane_change_cooldown
        self.reason = ""

    def reset(self):
        """Reset per-episode state."""
        self.steps_since_lane_change = self.lane_change_cooldown
        self.reason = ""

    def _parse(self, state: np.ndarray):
        """Decode the observation into ego lane/speed and neighbours in meters."""
        obs = state.reshape(5, 5)
        y_range = LANE_WIDTH * self.lanes_count
        ego_lane = int(np.clip(round(obs[0, 2] * y_range / LANE_WIDTH), 0, self.lanes_count - 1))
        ego_speed = obs[0, 3] * VX_RANGE
        others = []
        for presence, x, y, vx, _ in obs[1:]:
            if presence < 0.5:
                continue
            dy = y * y_range
            others.append({"dx": x * X_RANGE, "lane": int(round(dy / LANE_WIDTH)), "dvx": vx * VX_RANGE})
        return ego_lane, ego_speed, others

    @staticmethod
    def _front(others, lane_offset):
        """Closest car ahead in a lane: (bumper gap in m, TTC in s)."""
        ahead = [o for o in others if o["lane"] == lane_offset and o["dx"] > 0]
        if not ahead:
            return np.inf, np.inf
        car = min(ahead, key=lambda o: o["dx"])
        gap = car["dx"] - CAR_LENGTH
        closing = -car["dvx"]
        ttc = gap / closing if closing > 0.1 else np.inf
        return gap, ttc

    def _lane_free(self, others, ego_lane, lane_offset, front=20.0, back=10.0):
        """True if the lane exists and no visible car is inside the gap we would merge into."""
        if not 0 <= ego_lane + lane_offset < self.lanes_count:
            return False
        for o in others:
            if o["lane"] != lane_offset:
                continue
            # a faster car coming from behind needs a larger gap
            back_needed = back + 2.0 * max(o["dvx"], 0.0)
            if -back_needed < o["dx"] < front:
                return False
        return True

    def _change(self, action, reason):
        self.steps_since_lane_change = 0
        self.reason = reason
        return action

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> int:
        """
        Pick an action from the rules; `self.reason` describes the rule that fired.

        Args:
            state (np.ndarray): Flattened (25,) Kinematics observation.
            evaluate (bool): Unused, kept for agent API compatibility.

        Returns:
            int: DiscreteMetaAction index.
        """
        ego_lane, ego_speed, others = self._parse(state)
        gap, ttc = self._front(others, 0)
        self.steps_since_lane_change += 1
        cooled = self.steps_since_lane_change > self.lane_change_cooldown

        if gap < self.danger_gap or ttc < self.danger_ttc:
            if self._lane_free(others, ego_lane, -1):
                return self._change(self.LANE_LEFT, f"danger {gap:.0f} m ahead -> dodge left")
            if self._lane_free(others, ego_lane, +1):
                return self._change(self.LANE_RIGHT, f"danger {gap:.0f} m ahead -> dodge right")
            self.reason = f"danger {gap:.0f} m ahead, no free lane -> brake"
            return self.SLOWER

        if gap < self.follow_gap or ttc < self.follow_ttc:
            if cooled and self._lane_free(others, ego_lane, -1):
                return self._change(self.LANE_LEFT, f"slow car {gap:.0f} m ahead -> overtake left")
            self.reason = f"following car {gap:.0f} m ahead"
            return self.IDLE

        if cooled and self._lane_free(others, ego_lane, +1, front=30.0, back=15.0):
            return self._change(self.LANE_RIGHT, "road clear -> back to right lane")

        if ego_speed < self.top_speed:
            self.reason = f"road clear -> speed up ({ego_speed * 3.6:.0f} km/h)"
            return self.FASTER
        self.reason = "cruising"
        return self.IDLE


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser(description='Run the rule-based baseline')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=3, help='Number of highway lanes')
    parser.add_argument('--density', type=float, default=1.0, help='Traffic density (vehicles_density)')
    parser.add_argument('--duration', type=float, default=40,
                        help='Episode duration in seconds (>40 enables the endless highway)')
    parser.add_argument('--no_render', action='store_true')
    return parser.parse_args()


def main(args):
    """Evaluate the policy with the same env config and seeding as evaluate.py."""
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': args.lanes,
        'vehicles_density': args.density,
        'duration': args.duration,
        'ego_spacing': 1.5,
    }
    env = gymnasium.make("highway-v0", config=env_config, render_mode=None if args.no_render else 'human')
    env = make_long_episode(env, args.duration)
    policy = RuleBasedPolicy(lanes_count=args.lanes)

    returns, crashes = [], []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode - 1)
        state = state.reshape(-1)
        policy.reset()
        done = truncated = False
        episode_return, info = 0.0, {}
        while not (done or truncated):
            action = policy.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            state = state.reshape(-1)
            episode_return += reward
        crashed = bool(info.get("crashed", done and not truncated))
        returns.append(episode_return)
        crashes.append(crashed)
        print(f"Episode {episode:3d} | Return: {episode_return:6.2f} | Crash: {crashed}")

    print("-" * 50)
    print(f"Mean Return:  {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Crash Rate:   {np.mean(crashes) * 100:.1f}%")
    env.close()


if __name__ == "__main__":
    main(parse_args())
