class Environment:
    # Define actions
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, width, height, obstacles, start_pos, target_pos):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.robot_pos = start_pos  # Renamed from current_pos

    def is_valid_position(self, pos):
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if pos in self.obstacles:
            return False
        return True

    def get_sensor_readings(self):
        """
        Simulates simple proximity sensors.
        Returns a list of 4 boolean values indicating if the adjacent cell
        in each direction (up, down, left, right) is an obstacle or boundary.
        """
        px, py = self.robot_pos

        # Check Up: (px, py - 1)
        is_obstacle_up = not self.is_valid_position((px, py - 1))
        # Check Down: (px, py + 1)
        is_obstacle_down = not self.is_valid_position((px, py + 1))
        # Check Left: (px - 1, py)
        is_obstacle_left = not self.is_valid_position((px - 1, py))
        # Check Right: (px + 1, py)
        is_obstacle_right = not self.is_valid_position((px + 1, py))

        return [is_obstacle_up, is_obstacle_down, is_obstacle_left, is_obstacle_right]

    def reset(self):
        self.robot_pos = self.start_pos
        return self.get_state() # Return new state format

    def get_state(self):
        # State includes robot's position and sensor readings
        return (self.robot_pos, self.get_sensor_readings())

    def step(self, action):
        current_pos = self.robot_pos
        next_pos = current_pos

        if action == self.ACTION_UP:
            next_pos = (current_pos[0], current_pos[1] - 1)
        elif action == self.ACTION_DOWN:
            next_pos = (current_pos[0], current_pos[1] + 1)
        elif action == self.ACTION_LEFT:
            next_pos = (current_pos[0] - 1, current_pos[1])
        elif action == self.ACTION_RIGHT:
            next_pos = (current_pos[0] + 1, current_pos[1])

        collision = False
        if self.is_valid_position(next_pos):
            self.robot_pos = next_pos
        else:
            # Invalid move, robot stays in current_pos
            collision = True
            # next_pos remains current_pos for reward calculation if needed,
            # but self.robot_pos is what matters for state.

        done = False
        reward = 0

        if self.robot_pos == self.target_pos:
            reward = 100  # Reached target
            done = True
        elif collision:
            reward = -10  # Collided with obstacle or boundary
            # No explicit done = True for collision unless specified,
            # but often collisions might end an episode in some RL setups.
            # For now, let's make collision also mean 'done' as per instruction:
            # "done is True if self.robot_pos == self.target_pos or if there's a collision."
            done = True
        else:
            reward = -1  # Penalty for each step

        return self.get_state(), reward, done

if __name__ == '__main__':
    # Example usage:
    env_width = 10
    env_height = 10
    env_obstacles = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    env_start_pos = (0,0)
    env_target_pos = (9,9)

    environment = Environment(env_width, env_height, env_obstacles, env_start_pos, env_target_pos)

    print(f"Initial state: {environment.get_state()}")

    # Test moving right
    state, reward, done = environment.step(Environment.ACTION_RIGHT)
    print(f"Action: RIGHT, State: {state}, Reward: {reward}, Done: {done}")

    # Test moving into an obstacle (e.g. from (1,2) try to move to (2,2))
    environment.robot_pos = (1,2) # Manually set position for testing
    print(f"Manually set state for collision test: {environment.get_state()}")
    state, reward, done = environment.step(Environment.ACTION_RIGHT) # Move into obstacle (2,2)
    print(f"Action: RIGHT (into obstacle), State: {state}, Reward: {reward}, Done: {done}")

    # Test reset
    environment.reset()
    print(f"State after reset: {environment.get_state()}")

    # Test reaching target
    # For simplicity, let's assume the robot is one step away from the target
    environment.robot_pos = (9,8)
    print(f"Manually set state for target test: {environment.get_state()}")
    state, reward, done = environment.step(Environment.ACTION_DOWN) # Move to target (9,9)
    print(f"Action: DOWN (to target), State: {state}, Reward: {reward}, Done: {done}")

    # Test sensor readings from start
    environment.reset()
    print(f"Sensor readings at start {env_start_pos}: {environment.get_sensor_readings()}")
    # Manually move robot near an obstacle
    environment.robot_pos = (1,2)
    print(f"Sensor readings at {environment.robot_pos} (near obstacle (2,2)): {environment.get_sensor_readings()}")
    environment.robot_pos = (2,1) # Adjacent to (2,2)
    print(f"Sensor readings at {environment.robot_pos} (adj to obstacle (2,2)): {environment.get_sensor_readings()}")

    # Test out of bounds
    environment.robot_pos = (0,0)
    print(f"State before moving out of bounds: {environment.get_state()}")
    state, reward, done = environment.step(Environment.ACTION_UP) # Try to move up from (0,0)
    print(f"Action: UP (out of bounds), State: {state}, Reward: {reward}, Done: {done}")
    print(f"Robot position after trying to move out of bounds: {environment.robot_pos}")

    environment.robot_pos = (9,9) # Target
    print(f"State before moving out of bounds from target: {environment.get_state()}")
    state, reward, done = environment.step(Environment.ACTION_RIGHT) # Try to move right from (9,9)
    print(f"Action: RIGHT (out of bounds), State: {state}, Reward: {reward}, Done: {done}")
    print(f"Robot position after trying to move out of bounds from target: {environment.robot_pos}")
