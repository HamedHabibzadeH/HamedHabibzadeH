import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(state_dim,))
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim) # No activation, raw values/logits

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.state_fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.action_fc1 = tf.keras.layers.Dense(128, activation='relu')

        self.concat_layer = tf.keras.layers.Concatenate()
        self.combined_fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        state, action_indices = inputs
        action_one_hot = tf.one_hot(tf.squeeze(action_indices, axis=-1), depth=self.action_dim)

        state_out = self.state_fc1(state)
        state_out = self.state_fc2(state_out)
        action_out = self.action_fc1(action_one_hot)

        concat = self.concat_layer([state_out, action_out])
        q_value = self.combined_fc1(concat)
        return self.output_layer(q_value)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, 1), dtype=np.int32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0: # Avoid error if buffer is empty
            return None
        actual_batch_size = min(batch_size, self.size)
        idxs = np.random.choice(self.size, actual_batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action_value=None, # max_action_value not used for discrete
                 learning_rate_actor=0.001, learning_rate_critic=0.002,
                 gamma=0.99, tau=0.005, replay_buffer_size=50000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor Network and Target Actor Network
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights()) # Initialize target weights

        # Critic Network and Target Critic Network
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.set_weights(self.critic.get_weights()) # Initialize target weights

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_critic)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, action_dim)

    def select_action(self, state, exploration_noise=0.1):
        # state is a flat numpy array
        if state.ndim == 1: # Ensure state is batch-like (1, state_dim)
            state = np.expand_dims(state, axis=0)

        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action_logits = self.actor(state_tensor) # Output from actor are logits

        if np.random.rand() < exploration_noise:
            return np.random.randint(self.action_dim) # Explore: random action index
        else:
            return np.argmax(action_logits.numpy()[0]) # Exploit: action with highest logit

    def store_experience(self, state, action, reward, next_state, done):
        # Assuming state and next_state are already flat numpy arrays
        self.replay_buffer.store(state, action, reward, next_state, done)

    def update_target_networks(self):
        # Soft update for Actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        # Soft update for Critic
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def train(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        if batch is None: # Buffer might be empty or too small
            return

        states, actions, rewards, next_states, dones = batch

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32) # Action indices
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Critic Update
        with tf.GradientTape() as tape:
            # Get next actions (indices) from Target Actor's logits
            next_action_logits = self.target_actor(next_states)
            best_next_actions = tf.argmax(next_action_logits, axis=1)
            best_next_actions = tf.expand_dims(best_next_actions, axis=-1) # Ensure shape (batch, 1) for critic

            # Get Q-values from Target Critic
            target_q_values = self.target_critic([next_states, tf.cast(best_next_actions, dtype=tf.int32)])

            # Calculate target: y = R + gamma * Q_target(S', A'_target) * (1 - D)
            y = rewards + self.gamma * target_q_values * (1 - dones)

            # Current Q-values from Critic for (S, A) from batch
            current_q_values = self.critic([states, actions])

            # Critic loss
            critic_loss = tf.keras.losses.MSE(y, current_q_values)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Actor Update
        with tf.GradientTape() as tape:
            # Get action logits from Actor for current states
            predicted_action_logits = self.actor(states)
            # Determine actions actor would take (indices)
            predicted_actions_indices = tf.argmax(predicted_action_logits, axis=1)
            predicted_actions_indices = tf.expand_dims(predicted_actions_indices, axis=-1) # Shape for critic

            # Calculate Q-values for these (S, A_actor) using the Critic
            # The gradient will flow from critic's output back to actor's parameters
            q_values_for_actor_loss = self.critic([states, tf.cast(predicted_actions_indices, dtype=tf.int32)])

            # Actor loss: -mean(Q_critic(S, A_actor(S)))
            actor_loss = -tf.math.reduce_mean(q_values_for_actor_loss)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update Target Networks
        self.update_target_networks()

    def save_weights(self, path_prefix):
        self.actor.save_weights(path_prefix + '_actor.weights.h5')
        self.critic.save_weights(path_prefix + '_critic.weights.h5')
        self.target_actor.save_weights(path_prefix + '_target_actor.weights.h5')
        self.target_critic.save_weights(path_prefix + '_target_critic.weights.h5')
        print(f"Agent weights saved with prefix: {path_prefix}")

    def load_weights(self, path_prefix):
        try:
            self.actor.load_weights(path_prefix + '_actor.weights.h5')
            self.critic.load_weights(path_prefix + '_critic.weights.h5')
            self.target_actor.load_weights(path_prefix + '_target_actor.weights.h5')
            self.target_critic.load_weights(path_prefix + '_target_critic.weights.h5')
            print(f"Agent weights loaded from prefix: {path_prefix}")
        except Exception as e:
            print(f"Error loading weights: {e}. Ensure models are built or weights files exist.")


if __name__ == '__main__':
    state_dim_env = 6
    action_dim_env = 4

    # Instantiate DDPGAgent
    agent = DDPGAgent(state_dim_env, action_dim_env)
    print("DDPGAgent instantiated.")

    # Build networks by calling them once (optional, as they are built on first call in train/select_action)
    # Actor
    dummy_state_np = np.random.rand(1, state_dim_env).astype(np.float32)
    _ = agent.actor(tf.convert_to_tensor(dummy_state_np))
    _ = agent.target_actor(tf.convert_to_tensor(dummy_state_np))
    # Critic
    dummy_action_np = np.random.randint(0, action_dim_env, size=(1,1))
    _ = agent.critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])
    _ = agent.target_critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])
    print("Agent networks built.")


    # Test select_action
    print("\nTesting select_action...")
    dummy_state = np.random.rand(state_dim_env).astype(np.float32)
    action = agent.select_action(dummy_state, exploration_noise=0.5)
    print(f"  Selected action: {action} (exploration noise = 0.5)")
    action = agent.select_action(dummy_state, exploration_noise=0.0)
    print(f"  Selected action: {action} (exploration noise = 0.0)")

    # Store dummy experiences
    print("\nStoring dummy experiences...")
    for i in range(10): # Store enough for a small batch
        s = np.random.rand(state_dim_env).astype(np.float32)
        a = agent.select_action(s) # Use agent's policy to select action
        r = np.random.rand()
        s_prime = np.random.rand(state_dim_env).astype(np.float32)
        d = np.random.choice([True, False])
        agent.store_experience(s, a, r, s_prime, d)
        if i < 3: # Print first few
             print(f"  Stored: s=[{s[0]:.2f}..], a={a}, r={r:.2f}, s'=[{s_prime[0]:.2f}..], d={d}")
    print(f"Buffer size: {agent.replay_buffer.size}")

    # Test train method
    batch_size_train_test = 4
    if agent.replay_buffer.size >= batch_size_train_test:
        print(f"\nTesting train method with batch_size={batch_size_train_test}...")
        agent.train(batch_size_train_test)
        print("  Train method executed.")

        # Test saving weights
        print("\nTesting save_weights...")
        agent.save_weights('./test_ddpg_model_weights')

        # Test loading weights (into a new agent instance for a cleaner test)
        print("\nTesting load_weights...")
        agent_loaded = DDPGAgent(state_dim_env, action_dim_env)
        # Build models of new agent before loading (critical for Keras)
        _ = agent_loaded.actor(tf.convert_to_tensor(dummy_state_np))
        _ = agent_loaded.target_actor(tf.convert_to_tensor(dummy_state_np))
        _ = agent_loaded.critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])
        _ = agent_loaded.target_critic([tf.convert_to_tensor(dummy_state_np), tf.convert_to_tensor(dummy_action_np, dtype=tf.int32)])
        agent_loaded.load_weights('./test_ddpg_model_weights')
        print("  load_weights executed.")

    else:
        print(f"\nNot enough samples in buffer ({agent.replay_buffer.size}) to test train method with batch_size={batch_size_train_test}.")

    print("\nMain block test execution completed.")
