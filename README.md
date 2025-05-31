# DDPG Robot Navigation in a Grid World

This project implements a Deep Deterministic Policy Gradient (DDPG) agent to train a robot to navigate a 2D grid environment with obstacles and reach a target position.

## Project Structure

*   `environment.py`: Defines the 2D grid environment, robot kinematics, actions, rewards, and obstacle handling.
*   `ddpg_agent.py`: Implements the DDPG agent, including Actor and Critic networks, target networks, replay buffer, and learning logic.
*   `train.py`: Script for training the DDPG agent in the environment. It saves the trained model weights.
*   `test_agent.py`: Script for loading a trained model and testing its performance by running it in the environment and visualizing its path.

## Dependencies

*   Python (3.7+)
*   TensorFlow (2.x)
*   NumPy

You can install the Python dependencies using pip:
```bash
pip install tensorflow numpy
```

## How to Run

1.  **Train the Agent**:
    Open your terminal and run the training script:
    ```bash
    python train.py
    ```
    This will train the agent for a predefined number of episodes (default: 1000). After training, the model weights will be saved as:
    *   `ddpg_model_actor.weights.h5`
    *   `ddpg_model_critic.weights.h5`
    *   `ddpg_model_target_actor.weights.h5`
    *   `ddpg_model_target_critic.weights.h5`

2.  **Test and Visualize the Trained Agent**:
    After training is complete and the model files are saved, run the testing script:
    ```bash
    python test_agent.py
    ```
    This will load the trained weights, run the agent for one episode in the environment without exploration noise, print the path taken, total reward, and a text-based visualization of the path on the grid.

## Environment Configuration

The environment is defined in `environment.py`. You can modify its parameters in both `train.py` and `test_agent.py` when creating the `Environment` instance:
*   `width`, `height`: Dimensions of the grid.
*   `obstacles`: A list of `(x, y)` tuples representing obstacle locations.
*   `start_pos`: An `(x, y)` tuple for the robot's starting position.
*   `target_pos`: An `(x, y)` tuple for the target position.

Example:
```python
env = Environment(width=10, height=10, obstacles=[(2,2), (3,3)], start_pos=(0,0), target_pos=(9,9))
```

## Agent Hyperparameters

Key hyperparameters for the DDPG agent can be found and modified in:
*   `ddpg_agent.py` (within the `DDPGAgent` class `__init__` defaults): learning rates, gamma (discount factor), tau (soft update rate), replay buffer size.
*   `train.py`: number of episodes, max steps per episode, batch size, exploration noise parameters.

Tuning these parameters can affect the agent's learning performance and final path quality.
