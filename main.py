"""Grid world environment for reinforcement learning."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import typing
import math
import copy

# This MDP is completely specified & so V* & Q* will be solved exactly 
# with Dynamic Porgramming.


class GridWorld:
    """Grid world environment."""

    def __init__(self):
        """Initialize grid world environment."""

        # Create grid world environment
        u = 20
        l = 20
        self.number_of_rows = random.randint(l, u)
        self.number_of_columns = random.randint(l, u)
        self.grid_size = (self.number_of_rows, self.number_of_columns)
        self.number_of_states = self.number_of_rows * self.number_of_columns
        self.number_of_actions = 4
        # self.start_state = (0, 0)
        self.start_state = (random.randint(0, self.number_of_rows - 1), random.randint(0, self.number_of_columns - 1))
        self.state = self.start_state
        # self.terminal_states = [(self.number_of_rows - 1, self.number_of_columns - 1)]
        self.terminal_states = [(random.randint(0, self.number_of_rows - 1), random.randint(0, self.number_of_columns - 1))]
        min_number_of_obstacles = math.floor(self.number_of_states * 0.1)
        max_number_of_obstacles = math.floor(self.number_of_states * 0.2)
        self.number_of_obstacles = random.randint(min_number_of_obstacles,
                                                  max_number_of_obstacles)
        self.obstacles = []
        for i in range(self.number_of_obstacles):
            obstacle = (random.randint(0, self.number_of_rows - 1), random.randint(0, self.number_of_columns - 1))
            if obstacle not in self.terminal_states and obstacle != self.start_state:
                self.obstacles.append(obstacle)
        self.rewards = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.rewards[state] = 0
        for terminal_state in self.terminal_states:
            self.rewards[terminal_state] = 1
        self.policy: dict[tuple[int, int], dict[int, float]] = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.policy[state] = {}
                for action in range(self.number_of_actions):
                    self.policy[state][action] = 1 / self.number_of_actions
        self.state_transition_probabilities = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.state_transition_probabilities[state] = {}
                for action in range(self.number_of_actions):
                    next_state = self._get_next_state(action, state)
                    probability = 1
                    self.state_transition_probabilities[state][action] = {
                        'next state': next_state,
                        'probability': probability}
                    
        self.value_function = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.value_function[state] = 0

        self.fig, self.ax = plt.subplots()
        self.im =self.ax.imshow(np.zeros(self.grid_size))

        self._print_environment()
        self.iterative_policy_evaluation()
        self.plot_grid_world()
        # self.policy_improvement()
        # self.plot_grid_world()

    def _print_environment(self) -> None:
        """Print all the data related to the environment."""

        # np.set_printoptions(threshold=np.inf) # type: ignore

        print(f"Initialised environment of size "
              f"{self.number_of_rows} {self.number_of_columns}.")
        print(f"Created {self.number_of_states} states.")
        print(f"Every state is equivalent & shares the same actions, "
              f"for a total of {self.number_of_actions} actions per state.")
        print(f"Agent begins at {self.start_state} & terminates "
              f"at {self.terminal_states}.")
        print(f"Created {self.number_of_obstacles} obstacles.")
        # print(f"\nCreated rewards function.")
        # for state in self.rewards:
        #     print(f"state: {state}  reward: {self.rewards[state]}.")
        # print(f"\nCreated equiprobable policy.")
        # for state, policy_dictionary in self.policy.items():
        #     print(f"state {state}")
        #     for action in policy_dictionary:
        #         print(f"action {action} with probability {policy_dictionary[action]}")
        # print(f"\nCreated state transition probabilities.")
        # for state, action_dictionary in self.state_transition_probabilities.items():
        #     print(f"state: {state}.")
        #     for action, state_transition_dictionary in action_dictionary.items():
        #         print(f"action: {action}.")
        #         next_state = state_transition_dictionary['next state']
        #         probability = state_transition_dictionary['probability']
        #         print(f"next state {next_state} with probability {probability}"
        #             f" & rewawrd {self.rewards[next_state]}.")
                
    def plot_grid_world(self):
        """Plot the value function for the GridWorld."""

        # Plot grid world
        grid = np.zeros(self.grid_size)
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                grid[row][column] = self.value_function[(row, column)]
        for terminal_state in self.terminal_states:
            grid[terminal_state] = 1
        for obstacle in self.obstacles:
            grid[obstacle] = -1

        fix, ax = plt.subplots()
        ax.imshow(grid)
        plt.show()

    def _valid_action(self, row: int, column: int) -> bool:
        """Check if action is valid."""

        # Check if row is valid
        if row < 0 or row >= self.grid_size[0]:
            return False
        
        # Check if column is valid
        if column < 0 or column >= self.grid_size[1]:
            return False
        
        # TODO Check if obstacle
        if (row, column) in self.obstacles:
            return False

        # Action is valid
        return True

    def _get_next_state(self, action: int, 
                        state: tuple[int, int]|None = None) -> tuple[int, int]:
        """Get the next state given an action."""

        # Get the current state
        row, column = state or self.state
        
        # Take action
        match action:
            case 0: # Up
                # Check if action is valid.
                if self._valid_action(row - 1, column):
                    row -= 1
            case 1: # Left
                # Check if action is valid.
                if self._valid_action(row, column - 1):
                    column -= 1
            case 2: # Right
                # Check if action is valid.
                if self._valid_action(row, column + 1):
                    column += 1
            case 3: # Down
                # Check if action is valid.
                if self._valid_action(row + 1, column):
                    row += 1
            case _: # Invalid action
                raise ValueError(f"Invalid action: {action}")
            
        # Return next state.
        return row, column
    
    def _get_reward(self, state: tuple[int, int]) -> int:
        """Get reward for a given state."""

        # Get reward for state
        return self.rewards[state]
            
    def _check_if_done(self, state: tuple[int, int]) -> bool:
        """Check if the episode is done."""

        # Check if terminal state.
        if state in self.terminal_states:
            return True
        else:
            return False

    def reset(self) -> None:
        """Reset environment to start state."""

        self.state = self.start_state
    
    def bellman_equation_update_rule(self, state: tuple[int, int],
                                     gamma: float = 1) -> float:
        """Perform a bellman equation update for the state value function."""

        state_value = 0

        for action in range(self.number_of_actions):
            next_state = self._get_next_state(action, state)

            policy_action_probability = self.policy[state][action]
            transition_probability = self.state_transition_probabilities[state][action]['probability']

            next_state_reward = self.rewards[next_state]
            next_state_value = self.value_function[next_state]

            state_value += policy_action_probability * transition_probability * (next_state_reward + gamma  * next_state_value)

        return state_value


    def iterative_policy_evaluation(self, theta: float = 0.1):
        """Evaluate the value function for every state in the state set
        following policy pi."""

        # Initialise state-value function values (V) to zero, 
        # this can be done arbitrarily,
        # however, it is necessary for the terminal states to be zero, 
        # hence all states are set to zero.

        print("Evaluating policy with iterative policy evaluation.")

        while True:
            delta = 0
            for row in range(self.grid_size[0]):
                for column in range(self.grid_size[1]):
                    state = (row, column)
                    v = copy.deepcopy(self.value_function[state])
                    self.value_function[state] = self.bellman_equation_update_rule(state)
                    # print(f"state value {self.value_function[state]}")
                    delta = max(delta, abs(v - self.value_function[state]))
            if delta < theta:
                print(f"policy evaluation converged at {delta}.")
                print("Starting Policy Improvement.")
                # self.policy_improvement()
                break
            print(f"Continueing with delta at {delta}. Theta at {theta}")

        print("policy evaluation:\n")
        for state in self.value_function:
            print(f"{state} : {self.value_function[state]}")

    def policy_improvement(self, gamma: float = 0.9):
        """Determine an improved policy pi' from the value function of the old
        policy pi."""

        policy_stable = True

        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)

                state_actions_dict = self.state_transition_probabilities[state]
                action_values = []
                old_policy = copy.deepcopy(self.policy[state])

                # Get the value 
                for action in state_actions_dict.keys():
                    transition_probability = self.state_transition_probabilities[state][action]['probability']

                    next_state = self._get_next_state(action, state)
                    next_state_reward = self._get_reward(next_state)
                    next_state_value_function = self.value_function[next_state]
                    
                    action_values.append(transition_probability * (next_state_reward + gamma * next_state_value_function))
                    # actions_values.append(self.value_function[next_state])

                best_action_value = np.max(action_values)
                best_action_indices = np.where(action_values == best_action_value)
                for action in range(self.number_of_actions):
                    if action in best_action_indices[0]:
                        self.policy[state][action] = 1 / len(best_action_indices[0])
                    else:
                        self.policy[state][action] = 0

                if old_policy != self.policy[state]:
                    policy_stable = False
                #     print(state)
                #     print(action_values)
                #     print(best_action_indices[0])
                #     print(self.policy[state])

                # print(old_policy)
                # print(f"{self.policy[state]}")
                # print(f"{old_policy != self.policy[state]}\n")

        # Check if the policy is stable.
        if policy_stable:
            print("Policy stable. Stopping Policy Iteration.")
        else:
            print("Policy unstable. Evaluation Value Function for new policy.")
            # self.iterative_policy_evaluation()

if __name__ == '__main__':
    """BEGIN"""
    
    environment = GridWorld()
