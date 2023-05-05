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
        self.start_state = (0, 0)
        # self.start_state = (random.randint(0, self.number_of_rows - 1), random.randint(0, self.number_of_columns - 1))
        self.state = self.start_state
        self.terminal_states = [(self.number_of_rows - 1, self.number_of_columns - 1)]
        # self.terminal_states = [(random.randint(0, self.number_of_rows - 1), random.randint(0, self.number_of_columns - 1))]
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
        for obstacle in self.obstacles:
            self.rewards[obstacle] = -1
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
                    if next_state == state:
                        probability = 0
                    else:
                        probability = 1
                    self.state_transition_probabilities[state][action] = {
                        'next state': next_state,
                        'probability': probability}
                    
        self.value_function = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.value_function[state] = 0
        self.value_function_history = []

        self._print_environment()
        self.policy_iteration()
        self.plot_grid_world()

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

        fig, ax = plt.subplots()
        grid = np.zeros(self.grid_size)
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                grid[row][column] = self._get_state_value(state)
                for terminal_state in self.terminal_states:
                    grid[terminal_state] = 1
                for obstacle in self.obstacles:
                    grid[obstacle] = -1
                match self._get_policy_action(state):
                    case 0:
                        arrow_direction = (column, row - 1)
                    case 1:
                        arrow_direction = (column - 1, row)
                    case 2:
                        arrow_direction = (column + 1, row)
                    case 3:
                        arrow_direction = (column, row + 1)
                    case _:
                        raise ValueError("Invalid action.")
                print(f"{state} : {self._get_policy_action(state)}")
                if state in self.terminal_states or state in self.obstacles or state == self.start_state:
                    continue
                ax.annotate("", xy=arrow_direction,
                                    xytext=(column, row),
                                    arrowprops=dict(arrowstyle="->",
                                    color="black"),
                                    size=15)

        ax.imshow(grid)
        # for row in range(self.number_of_rows):
        #     for column in range(self.number_of_columns):
                
        # ax.colorbar()

        # if len(self.value_function_history) == 1:
        #     for row in range(self.number_of_rows):
        #         for column in range(self.number_of_columns):
        #             grid[row][column] = self.value_function[(row, column)]
        #             for terminal_state in self.terminal_states:
        #                 grid[terminal_state] = 1
        #             for obstacle in self.obstacles:
        #                 grid[obstacle] = -1
        #     plt.imshow(grid)
        # else:
        #     fig, axes = plt.subplots(1, len(self.value_function_history), figsize=(15, 5 * len(self.value_function_history)))
        #     for i, value_function in enumerate(self.value_function_history):
        #         for row in range(self.number_of_rows):
        #             for column in range(self.number_of_columns):
        #                 grid[row][column] = value_function[(row, column)]
        #         for terminal_state in self.terminal_states:
        #             grid[terminal_state] = 1
        #         for obstacle in self.obstacles:
        #             grid[obstacle] = -1

        #         axes[i].imshow(grid)
        #         axes[i].set_title("Policy Evaluation" if i % 2 == 0 else "Policy Improvement")
        plt.show()

    def _valid_action(self, row: int, column: int) -> bool:
        """Check if action is valid."""

        # Check if row is valid
        if row < 0 or row >= self.number_of_rows:
            return False
        
        # Check if column is valid
        if column < 0 or column >= self.number_of_columns:
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
    
    def _get_state_value(self, state: tuple[int, int]) -> float:
        """Get the value of a state."""

        # Get value of state
        return self.value_function[state]
    
    def _get_state_transition_probability(self, state: tuple[int, int],
                                          action: int) -> float:
        """Get the probability of a state transition."""

        # Get state transition probability.
        return self.state_transition_probabilities[state][action]['probability']
    
    def _get_policy_action_probability(self, state: tuple[int, int],
                                       action: int) -> float:
        """Get the probability of an action in the policy."""

        # Get policy action probability.
        return self.policy[state][action]
    
    def _get_policy_action(self, state: tuple[int, int]) -> int:
        """Get the action to take given the policy."""

        best_action = 0

        for action in range(self.number_of_actions):
            if self.policy[state][action] > self.policy[state][best_action]:
                best_action = action

        # Get policy action.
        return best_action
            
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

    def policy_iteration(self):
        """Policy Iteration to determine the Value Function."""

        self.iterative_policy_evaluation()
    
    def bellman_equation_update_rule(self, state: tuple[int, int],
                                     gamma: float = 1) -> float:
        """Perform a bellman equation update for the state value function."""

        state_value = 0

        for action in range(self.number_of_actions):
            next_state = self._get_next_state(action, state)
            policy_action_probability = self._get_policy_action_probability(state, action)
            transition_probability = self._get_state_transition_probability(state, action)
            next_state_reward = self._get_reward(next_state)
            next_state_value = self._get_state_value(next_state)
            state_value += policy_action_probability * transition_probability * (next_state_reward + gamma * next_state_value)

        return state_value

    def iterative_policy_evaluation(self, theta: float = 1):
        """Evaluate the value function for every state in the state set
        following policy pi."""

        print("Evaluating policy with iterative policy evaluation.")
        
        while True:
            delta = 0
            for row in range(self.number_of_rows):
                for column in range(self.number_of_columns):
                    state = (row, column)
                    if state in self.terminal_states:
                        continue
                    old_state_value: dict = copy.deepcopy(self.value_function[state])
                    self.value_function[state] = self.bellman_equation_update_rule(state)
                    delta = max(delta, abs(old_state_value - self.value_function[state]))
            if delta < theta:
                print(f"policy evaluation converged at {delta}.")
                print("Starting Policy Improvement.")
                self.value_function_history.append(copy.deepcopy(self.value_function))
                self.policy_improvement()
                break
            print(f"\rContinuing with Theta {theta} & delta {delta}", end = "")

    def policy_improvement(self, gamma: float = 1):
        """Determine an improved policy pi' from the value function of the old
        policy pi."""

        print("Improving policy.")

        policy_stable = True

        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)

                action_values: list[float] = []
                old_policy: dict = copy.deepcopy(self.policy[state])

                # Get the value 
                for action in range(self.number_of_actions):
                    transition_probability = self.state_transition_probabilities[state][action]['probability']

                    next_state = self._get_next_state(action, state)
                    next_state_reward = self._get_reward(next_state)
                    next_state_value_function = self._get_state_value(next_state)
                    
                    action_values.append(transition_probability * (next_state_reward + gamma * next_state_value_function))

                # Get the best action
                best_action = np.argmax(action_values)
                for action in range(self.number_of_actions):
                    if action == best_action:
                        self.policy[state][action] = 1 / len(action_values)
                    else:
                        self.policy[state][action] = 0

                # Check if policy is stable
                if old_policy != self.policy[state]:
                    policy_stable = False

        # Check if the policy is stable.
        if policy_stable:
            print("Policy stable. Stopping Policy Iteration.")
        else:
            print("Policy unstable. New policy found.")
            print("Evaluating Value Function for new policy.\n")
            self.iterative_policy_evaluation()

if __name__ == '__main__':
    """BEGIN"""
    
    environment = GridWorld()
