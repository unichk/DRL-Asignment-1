import gym
import importlib.util
import numpy as np
import random
import time
from IPython.display import clear_output
from itertools import product

class TaxiEnv(gym.Env):
    def __init__(self, fuel_limit = 5000, seed = 0):
        random.seed(seed)

        self.fuel_limit = fuel_limit

    def reset(self):  
        self.grid_size = random.randint(5, 20)
        self.all_cells = list(product(range(self.grid_size), range(self.grid_size)))

        self.locs = random.choices(self.all_cells, k = 4)
        self.passenger_loc, self.destination = random.choices(self.locs, k = 2)
        self.obstacles = set(random.choices(self.all_cells, k = random.randint(0, int(0.4 * self.grid_size * self.grid_size)))) - set(self.locs)

        self.taxi_row = random.randint(0, self.grid_size)
        self.taxi_col = random.randint(0, self.grid_size)
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        return self.get_state(), {}

    def get_state(self):
        taxi_row = self.taxi_row
        taxi_col = self.taxi_col
        
        obstacle_north = int(self.taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        
        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        
        destination_loc_north = int((taxi_row - 1, taxi_col) in self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) in self.destination)
        destination_loc_east  = int((taxi_row, taxi_col + 1) in self.destination)
        destination_loc_west  = int((taxi_row, taxi_col - 1) in self.destination)
        destination_loc_middle  = int((taxi_row, taxi_col) in self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle
        
        state = (taxi_row, taxi_col, self.locs[0][0], self.locs[0][1], self.locs[1][0], self.locs[1][1], self.locs[2][0], self.locs[2][1], self.locs[3][0], self.locs[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def step(self, action):
        self.current_fuel -= 1

        next_row, next_col = self.taxi_row, self.taxi_col
        if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        if action in [0, 1, 2, 3]:  
            if not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size) or (next_row, next_col) in self.obstacles:
                reward = -5.1
                if self.current_fuel <= 0:
                    return self.get_state(), reward -10, False, True, {}
                return self.get_state(), reward, False, False, {}
            reward = 0

        self.taxi_row, self.taxi_col = taxi_row, taxi_col = next_row, next_col

        if action == 4:
            if (taxi_row, taxi_col) == self.passenger_loc and not self.passenger_picked_up:
                self.passenger_picked_up = True 
            else:
                reward = -10
        elif action == 5:  
            if self.passenger_picked_up:
                if (taxi_row, taxi_col) == self.destination:
                    reward = 50
                    return self.get_state(), reward -0.1, True, False, {}
            self.passenger_picked_up = False
            reward = -10

        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)  
        if self.current_fuel <= 0:
            return self.get_state(), reward -10.1, False, True, {}  
        return self.get_state(), reward - 0.1, False, False, {}

    def render_env(self, taxi_pos, action = None, reward = None, fuel = None):
        print(f"action: {self.get_action_name(action)}")
        print(f"{reward = }")
        
        grid = np.full((self.grid_size, self.grid_size), '.')
        for obstacle in self.obstacles:
            grid[obstacle] = 'X'
        for i, c in enumerate(['R', 'G', 'Y', 'B']):
            grid[self.locs[i]] = c
        if not self.passenger_picked_up:
            grid[self.passenger_loc] = 'P'
        grid[self.destination] = 'D'
        if taxi_pos in self.locs:
            grid[taxi_pos] = grid[taxi_pos].item().lower()
        else:
            grid[taxi_pos] = 'T'
        print("\n".join(" ".join(s) for s in grid))
        clear_output(wait = True)
        time.sleep(0.2)

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = TaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    
    taxi_row, taxi_col, *_ = obs

    while not done:
        action = student_agent.get_action(obs)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"{obs = }")
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, *_ = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action = action, reward = reward, fuel = env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render = True)
    print(f"Final Score: {agent_score}")