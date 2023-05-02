"""Grid world environment for reinforcement learning."""

import numpy as np
import matplotlib.pyplot as plt
import random
import typing

# This MDP is completely specified & so V* & Q* will be solved exactly 
# with Dynamic Porgramming.


class GridWorld:
    """Grid world environment."""

    def __init__(self):
        """Initialize grid world environment."""

        # Create grid world environment
        self.number_of_rows = 10
        self.number_of_columns = 10
        self.grid_size = (self.number_of_rows, self.number_of_columns)
        self.number_of_states = self.number_of_rows * self.number_of_columns
        self.number_of_actions = 4
        self.start_state = (0, 0)
        self.state = self.start_state
        self.terminal_states = [(self.grid_size[0] - 1, self.grid_size[1] - 1)]
        min_number_of_obstacles = 10
        max_number_of_obstacles = 20
        self.number_of_obstacles = random.randint(min_number_of_obstacles,
                                                  max_number_of_obstacles)
        self.obstacles = [(random.randint(0, self.grid_size[0] - 1), 
                    random.randint(0, self.grid_size[1] - 1)) for obstacle 
                    in range(self.number_of_obstacles)]
        self.rewards = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.rewards[state] = 0
        for terminal_state in self.terminal_states:
            self.rewards[terminal_state] = 1
        self.policy: dict[tuple[int, int], dict] = {}
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                state = (row, column)
                self.policy[state] = {}
                for action in range(self.number_of_actions):
                    self.policy[state][action] = 1 / self.number_of_actions
                
        # self.policy: dict[tuple[int, int], int] = {}
        # for row in range(self.number_of_rows):
        #     for column in range(self.number_of_columns):
        #         state = (row, column)
        #         self.policy[state] = 0
        self.state_transition_probabilities: dict[tuple[int, int], dict[int, dict[str, typing.Union[tuple[int, int], float]]]] = {}
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

        self._print_environment()

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
        print(f"\nCreated rewards function.")
        for state in self.rewards:
            print(f"state: {state}  reward: {self.rewards[state]}.")
        print(f"\nCreated equiprobable policy.")
        for state, policy_dictionary in self.policy.items():
            print(f"state {state}")
            for action in policy_dictionary:
                print(f"action {action} with probability {policy_dictionary[action]}")
        print(f"\nCreated state transition probabilities.")
        for state, action_dictionary in self.state_transition_probabilities.items():
            print(f"state: {state}.")
            for action, state_transition_dictionary in action_dictionary.items():
                print(f"action: {action}.")
                next_state = state_transition_dictionary['next state']
                probability = state_transition_dictionary['probability']
                print(f"next state {next_state} with probability {probability}"
                    f" & rewawrd {self.rewards[next_state]}.")

    def _valid_action(self, row: int, column: int) -> bool:
        """Check if action is valid."""

        # Check if row is valid
        if row < 0 or row >= self.grid_size[0]:
            return False
        
        # Check if column is valid
        if column < 0 or column >= self.grid_size[1]:
            return False
        
        # TODO Check if obstacle

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
                                     V: dict,
                                     gamma: float = 1) -> float:
        """Perform a bellman equation update for the state value function."""

        for action in range(self.number_of_actions):
            next_state = self._get_next_state(action, state)

            policy_action = self.policy[state]
            transition_probability = self.state_transition_probabilities[state][action]['probability']

            next_state_reward = self.rewards[next_state]
            next_state_value = V[next_state]

        return 1


    def iterative_policy_evaluation(self, pi: dict, theta: float = 0.0001):
        """Iterative policy evaluation algorithm to find state-value function."""

        # Initialise state-value function values (V) to zero, 
        # this can be done arbitrarily,
        # however, it is necessary for the terminal states to be zero, 
        # hence all states are set to zero.
        V = {}
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                state = (row, column)
                V[state] = 0

        while True:
            delta = 0
            for row in range(self.grid_size[0]):
                for column in range(self.grid_size[1]):
                    state = (row, column)
                    v = V[state]
                    V[state] = self.bellman_equation_update_rule(state, V)
                    delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break

if __name__ == '__main__': 
    
    environment = GridWorld()


    # # Plot grid world
    # grid = np.zeros(environment.grid_size)
    # for terminal_state in environment.terminal_states:
    #     grid[terminal_state] = 1
    # for obstacle in environment.obstacles:
    #     grid[obstacle] = -1
    # plt.imshow(grid, cmap='gray')
    # plt.show()
