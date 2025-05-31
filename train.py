import numpy as np
import tensorflow as tf

from environment import Environment
from ddpg_agent import DDPGAgent # Assuming Actor, Critic, ReplayBuffer are in ddpg_agent.py or imported by it

def flatten_state(state_tuple):
    """Converts ((x,y), (s1,s2,s3,s4)) to [x, y, s1, s2, s3, s4]."""
    return np.array(list(state_tuple[0]) + list(state_tuple[1]), dtype=np.float32)

def main():
    # Training Parameters
    num_episodes = 1000
    max_steps_per_episode = 100
    batch_size = 64

    exploration_noise_start = 1.0
    exploration_noise_end = 0.01 # Adjusted from 0.1 to ensure more exploration initially
    # exploration_decay_rate = 0.995 # Per episode
    # Let's define decay such that it reaches exploration_noise_end in approx num_episodes
    # Or more simply, decay per episode:
    exploration_decay_steps = num_episodes * 0.8 # Decay over 80% of episodes
    exploration_decay_rate = np.exp(np.log(exploration_noise_end / exploration_noise_start) / exploration_decay_steps)


    # Initialize Environment
    env = Environment(
        width=10,
        height=10,
        obstacles=[(2,2), (3,3), (4,4), (5,5), (6,6)],
        start_pos=(0,0),
        target_pos=(9,9)
    )

    # State and Action Dimensions
    # State from env.get_state() is ((x,y), (s1,s2,s3,s4))
    # Flattened state: [x, y, s1, s2, s3, s4]
    state_dim = 2 + 4  # pos_dim + sensor_dim
    action_dim = 4     # Corresponds to env.ACTION_UP, ACTION_DOWN, etc.

    # Initialize Agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        # max_action_value is part of DDPGAgent constructor but not critical for discrete actions
        # It's often used to scale tanh output in continuous DDPG. We can pass a placeholder.
        max_action_value=1.0,
        learning_rate_actor=0.001,
        learning_rate_critic=0.002,
        gamma=0.99,
        tau=0.005,
        replay_buffer_size=50000
    )

    print(f"Starting training for {num_episodes} episodes...")
    print(f"Exploration noise will decay from {exploration_noise_start:.2f} to ~{exploration_noise_end:.2f} with rate {exploration_decay_rate:.5f}")

    exploration_noise = exploration_noise_start
    total_rewards_history = []

    for episode in range(num_episodes):
        current_state_tuple = env.reset()
        current_state_flat = flatten_state(current_state_tuple)
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(current_state_flat, exploration_noise)

            # Environment step returns: ( (next_pos), (next_sensor_readings) ), reward, done
            next_state_tuple, reward, done = env.step(action)
            next_state_flat = flatten_state(next_state_tuple)

            agent.store_experience(current_state_flat, action, reward, next_state_flat, done)

            # Train agent
            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            current_state_flat = next_state_flat
            episode_reward += reward

            if done:
                break

        # Decay exploration noise
        exploration_noise = max(exploration_noise_end, exploration_noise * exploration_decay_rate)
        # A simpler linear decay could also be used:
        # exploration_noise = exploration_noise_start - (exploration_noise_start - exploration_noise_end) * (episode / num_episodes)
        # exploration_noise = max(exploration_noise_end, exploration_noise)


        total_rewards_history.append(episode_reward)
        if (episode + 1) % 10 == 0: # Print every 10 episodes
            avg_reward = np.mean(total_rewards_history[-100:]) # Avg reward of last 100 episodes
            print(f"Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, AvgReward (last 100): {avg_reward:.2f}, Noise: {exploration_noise:.3f}")

    print("\nTraining completed.")

    # Save agent weights
    agent.save_weights('./ddpg_model')
    print("Agent weights saved to ./ddpg_model_actor.weights.h5, etc.")

if __name__ == '__main__':
    # Set random seeds for reproducibility (optional)
    # np.random.seed(42)
    # tf.random.set_seed(42)
    main()
