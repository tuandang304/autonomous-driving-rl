"""Endless-highway wrapper for long highway-v0 episodes."""

import gymnasium
import numpy as np

DEFAULT_DURATION = 40


class EndlessHighway(gymnasium.Wrapper):
    """
    Keeps highway-v0 playable for long episodes (e.g. 1000 s).

    The stock road is 10 km long and traffic is only spawned at reset, so after a
    few minutes the ego drives off the road (reward becomes 0) or alone. This wrapper:
    - extends every lane to `road_length` meters after reset;
    - moves vehicles that fell more than `behind` meters behind the ego to a free
      spot ahead, beyond the agent's perception range, keeping traffic density stable.
    Recycling uses its own RNG so the env's random stream is untouched.
    """

    def __init__(self, env: gymnasium.Env, road_length: float = 100_000.0, behind: float = 120.0,
                 ahead_min: float = 220.0):
        """
        __init__.

        Args:
            env (gymnasium.Env): highway-v0 environment.
            road_length (float): Lane length in meters after extension.
            behind (float): Distance behind the ego after which a vehicle is recycled.
            ahead_min (float): Minimum distance ahead of the ego for a recycled vehicle.
        """
        super().__init__(env)
        self.road_length = road_length
        self.behind = behind
        self.ahead_min = ahead_min
        self.rng = np.random.default_rng()
        self.spread = 1000.0

    def reset(self, **kwargs):
        """Reset, extend the road and measure the initial traffic spread."""
        obs, info = self.env.reset(**kwargs)
        seed = kwargs.get("seed")
        self.rng = np.random.default_rng(None if seed is None else seed + 10_000)
        road = self.env.unwrapped.road
        for lane in road.network.lanes_list():
            lane.end = lane.start + lane.direction * self.road_length
            lane.length = self.road_length
        ego = self.env.unwrapped.vehicle
        self.spread = max(300.0, max(v.position[0] for v in road.vehicles) - ego.position[0])
        return obs, info

    def step(self, action):
        """Step the env, then recycle vehicles left far behind."""
        result = self.env.step(action)
        self._recycle()
        return result

    def _recycle(self):
        uenv = self.env.unwrapped
        ego, road = uenv.vehicle, uenv.road
        origin, dest = ego.lane_index[0], ego.lane_index[1]
        n_lanes = len(road.network.graph[origin][dest])
        for v in road.vehicles:
            if v is ego or v.crashed or v.position[0] > ego.position[0] - self.behind:
                continue
            for _ in range(20):
                lane_id = int(self.rng.integers(n_lanes))
                x = ego.position[0] + self.ahead_min + self.rng.uniform(0.0, self.spread)
                if all(o is v or o.lane_index[2] != lane_id or abs(o.position[0] - x) > 25.0
                       for o in road.vehicles):
                    break
            lane_index = (origin, dest, lane_id)
            lane = road.network.get_lane(lane_index)
            v.position = lane.position(x, 0.0)
            v.heading = lane.heading_at(x)
            v.lane_index = v.target_lane_index = lane_index
            v.lane = lane
            v.speed = v.target_speed = float(self.rng.uniform(21.0, 25.0))
            v.history.clear()


def make_long_episode(env: gymnasium.Env, duration: float) -> gymnasium.Env:
    """
    Wrap with EndlessHighway only when the episode is longer than the default.

    The 40 s default stays bit-identical to the original setup used for evaluation.

    Args:
        env (gymnasium.Env): highway-v0 environment created with `duration` in its config.
        duration (float): Episode duration in seconds.

    Returns:
        gymnasium.Env: The original env, or the wrapped one for long episodes.
    """
    return EndlessHighway(env) if duration > DEFAULT_DURATION else env
