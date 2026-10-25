"""
Evaluation script for trained RL agents.

Usage:
    python evaluate.py
    python evaluate.py --weights weights/best_model.pth --episodes 10
"""

import gymnasium
import highway_env
import numpy as np
import argparse
from tqdm import tqdm

from src.utils import set_seed
from src.env.endless_highway import make_long_episode
from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='d3qn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='Enable action smoothing to prevent jittery lane changes')
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable action smoothing')
    parser.add_argument('--smooth_window', type=int, default=3,
                        help='Minimum steps between lane changes')
    parser.add_argument('--lanes', type=int, default=3,
                        help='Number of highway lanes')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Traffic density (vehicles_density)')
    parser.add_argument('--duration', type=float, default=40,
                        help='Episode duration in seconds (>40 enables the endless highway)')
    return parser.parse_args()


class ActionSmoother:
    """Prevents jittery lane changes but allows emergency maneuvers."""
    
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    
    def __init__(self, cooldown: int = 3):
        """
        __init__.
        
        Args:
            cooldown (int): Parameter.
        """
        self.cooldown = cooldown
        self.steps_since_lane_change = cooldown
        self.last_lane_action = None
        self.last_state = None
    
    def filter(self, action: int, state: np.ndarray = None) -> int:
        """Filter action to prevent jittery behavior, but allow emergency maneuvers."""
        is_lane_change = action in [self.LANE_LEFT, self.LANE_RIGHT]
        
        # Check for emergency situation (vehicle very close ahead)
        emergency = False
        if state is not None:
            state_matrix = state.reshape(5, 5)
            for i in range(1, 5):
                presence, x, y, vx, vy = state_matrix[i]
                if presence > 0.5:
                    # Vehicle in same lane, very close ahead
                    if abs(y) < 0.1 and 0 < x < 0.08:
                        emergency = True
                        break
                    # Vehicle approaching fast from behind in same lane
                    if abs(y) < 0.1 and -0.05 < x < 0 and vx > 0.1:
                        emergency = True
                        break
        
        if is_lane_change:
            # Always allow emergency lane changes
            if emergency:
                self.steps_since_lane_change = 0
                self.last_lane_action = action
                return action
            
            # Block lane change if cooldown not expired
            if self.steps_since_lane_change < self.cooldown:
                return self.IDLE
            
            # Block opposite lane change immediately after (prevents zigzag)
            if self.last_lane_action is not None:
                if (action == self.LANE_LEFT and self.last_lane_action == self.LANE_RIGHT) or \
                   (action == self.LANE_RIGHT and self.last_lane_action == self.LANE_LEFT):
                    if self.steps_since_lane_change < self.cooldown * 2:
                        return self.IDLE
            
            # Allow lane change
            self.steps_since_lane_change = 0
            self.last_lane_action = action
            return action
        else:
            self.steps_since_lane_change += 1
            return action
    
    def reset(self):
        """
        reset.
        """
        self.steps_since_lane_change = self.cooldown
        self.last_lane_action = None


def load_agent(algorithm: str, state_dim: int, action_dim: int, weights_path: str):
    """Load trained agent."""
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif algorithm == 'double_dqn':
        agent = DoubleDQNAgent(state_dim, action_dim)
    elif algorithm == 'dueling_dqn':
        agent = DuelingDQNAgent(state_dim, action_dim)
    elif algorithm == 'd3qn':
        agent = D3QNAgent(state_dim, action_dim)
    elif algorithm == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' not yet implemented")
    
    agent.load(weights_path)
    return agent


def evaluate(args):
    """Main evaluation loop."""
    set_seed(args.seed)
    
    # Environment config
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': args.lanes,
        'vehicles_density': args.density,
        'duration': args.duration,
        'ego_spacing': 1.5,
    }
    
    render_mode = None if args.no_render else 'human'
    env = make_long_episode(gymnasium.make("highway-v0", config=env_config, render_mode=render_mode), args.duration)
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    print(f"Loading agent from {args.weights}")
    agent = load_agent(args.algorithm, state_dim, action_dim, args.weights)
    
    print(f"Evaluating for {args.episodes} episodes...")
    use_smooth = args.smooth and not args.no_smooth
    
    # Action smoother
    smoother = ActionSmoother(cooldown=args.smooth_window) if use_smooth else None
    
    # Metrics
    returns = []
    lengths = []
    crashes = []
    
    # Use tqdm for progress bar (disable per-episode print if no_render)
    show_episodes = render_mode is not None
    episode_iter = tqdm(range(1, args.episodes + 1), desc="Evaluating", disable=False)
    
    for episode in episode_iter:
        state, _ = env.reset(seed=args.seed + episode - 1)
        state = state.reshape(-1)
        done, truncated = False, False
        
        if smoother:
            smoother.reset()
        
        episode_return = 0
        episode_steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            
            # Apply smoothing if enabled (pass state for emergency detection)
            if smoother:
                action = smoother.filter(action, state)
            
            state, reward, done, truncated, info = env.step(action)
            state = state.reshape(-1)
            
            if render_mode:
                env.render()
            
            episode_return += reward
            episode_steps += 1
            
            crashed = info.get("crashed", done and not truncated)
        
        returns.append(episode_return)
        lengths.append(episode_steps)
        crashes.append(crashed)
        
        # Update progress bar with current stats
        episode_iter.set_postfix({
            'return': f'{episode_return:.1f}',
            'crash': done,
            'avg': f'{np.mean(returns):.1f}'
        })
    
    # Summary
    print("-" * 50)
    print("EVALUATION SUMMARY")
    print(f"Mean Return:  {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Crash Rate:   {np.mean(crashes)*100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)