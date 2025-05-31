import numpy as np
import tensorflow as tf # Needed for DDPGAgent class structure, even if not directly used for ops here.

from environment import Environment
from ddpg_agent import DDPGAgent

def flatten_state(state_tuple):
    """Converts ((x,y), (s1,s2,s3,s4)) to [x, y, s1, s2, s3, s4]."""
    if state_tuple is None or state_tuple[0] is None or state_tuple[1] is None:
        # Handle cases where state might be incomplete, though env.step should always return full state.
        # This might happen if trying to flatten a None state if env.reset() or env.step() failed.
        # For robust error handling, one might raise an error or return a default error state.
        # Here, we'll print an error and return a zero array of expected shape if possible,
        # or raise error if state_dim isn't obvious.
        print("Error: Received an invalid state_tuple for flattening.")
        # Defaulting to a zero state of size 6 if this happens, though ideally it shouldn't.
        return np.zeros(6, dtype=np.float32)
    return np.array(list(state_tuple[0]) + list(state_tuple[1]), dtype=np.float32)

def visualize_path(env, path):
    """Creates a text-based visualization of the grid and the agent's path."""
    grid_vis = [['.' for _ in range(env.width)] for _ in range(env.height)]

    # Mark obstacles
    for obs_x, obs_y in env.obstacles:
        if 0 <= obs_y < env.height and 0 <= obs_x < env.width: # Check bounds for obstacles
            grid_vis[obs_y][obs_x] = 'X'

    # Mark start and target
    if 0 <= env.start_pos[1] < env.height and 0 <= env.start_pos[0] < env.width:
        grid_vis[env.start_pos[1]][env.start_pos[0]] = 'S'
    if 0 <= env.target_pos[1] < env.height and 0 <= env.target_pos[0] < env.width:
        grid_vis[env.target_pos[1]][env.target_pos[0]] = 'T'

    # Mark path
    if path:
        for i, (pos_x, pos_y) in enumerate(path):
            if 0 <= pos_y < env.height and 0 <= pos_x < env.width: # Check bounds for path points
                if grid_vis[pos_y][pos_x] == '.': # Don't overwrite S or T with path marker '*'
                    grid_vis[pos_y][pos_x] = '*'
                elif grid_vis[pos_y][pos_x] == 'S' and i > 0 : # Robot moved back to Start
                     grid_vis[pos_y][pos_x] = 'S*'
                elif grid_vis[pos_y][pos_x] == 'T' and (pos_x,pos_y) == env.target_pos: # Reached Target
                     grid_vis[pos_y][pos_x] = 'T*'


    print("\nPath Visualization:")
    for row in grid_vis:
        print(' '.join(row))
    print("\nLegend: S=Start, T=Target, X=Obstacle, *=Path")


def main():
    # Initialize Environment (same as in train.py)
    env = Environment(
        width=10,
        height=10,
        obstacles=[(2,2), (3,3), (4,4), (5,5), (6,6)],
        start_pos=(0,0),
        target_pos=(9,9)
    )

    state_dim = 2 + 4  # pos_dim + sensor_dim
    action_dim = 4

    # Initialize Agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action_value=1.0 # Placeholder, not critical for discrete actions here
    )

    # Build agent models by calling them once (essential before loading weights if not done in constructor)
    # This ensures the variables are created.
    dummy_state_np = np.random.rand(1, state_dim).astype(np.float32)
    _ = agent.actor(tf.convert_to_tensor(dummy_state_np))
    _ = agent.target_actor(tf.convert_to_tensor(dummy_state_np)) # Also build target networks
    dummy_action_np = np.random.randint(0, action_dim, size=(1,1))
    _ = agent.critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])
    _ = agent.target_critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])


    # Load saved weights
    model_path_prefix = './ddpg_model' # Must match the prefix used in train.py
    agent.load_weights(model_path_prefix)

    print("\nStarting testing phase with loaded weights...")

    current_state_tuple = env.reset()
    current_state_flat = flatten_state(current_state_tuple)

    episode_path = [env.start_pos] # Store sequence of (x,y) positions
    total_reward = 0
    max_steps_per_episode = 100 # Same as in training

    for step in range(max_steps_per_episode):
        action = agent.select_action(current_state_flat, exploration_noise=0.0) # Deterministic

        next_state_tuple, reward, done = env.step(action)
        next_state_flat = flatten_state(next_state_tuple)

        robot_pos = next_state_tuple[0] # This is the new self.robot_pos after env.step()
        episode_path.append(robot_pos)
        total_reward += reward
        current_state_flat = next_state_flat

        print(f"Step: {step+1}, Pos: {robot_pos}, Action: {action}, Reward: {reward:.2f}, Done: {done}")

        if done:
            break

    print("\nTesting finished.")
    print(f"Path taken: {episode_path}")
    print(f"Total reward: {total_reward:.2f}")

    visualize_path(env, episode_path)

if __name__ == '__main__':
    main()
