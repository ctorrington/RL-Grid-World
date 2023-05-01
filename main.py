"""Grid world environment for reinforcement learning."""

import numpy as np
import matplotlib.pyplot as plt
import random

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
        # self.rewards = np.zeros((self.number_of_rows,
        #                          self.number_of_columns,
        #                          self.number_of_actions))
        # for terminal_state in self.terminal_states:
        #     row, column = terminal_state
        #     self.rewards[row, column, :] = 1
        self.policy = np.zeros((self.number_of_rows,
                               self.number_of_columns,
                               self.number_of_actions))
        self.state_transitions = np.zeros((self.number_of_rows,
                                           self.number_of_columns,
                                           self.number_of_actions,
                                           self.number_of_rows,
                                           self.number_of_columns))
        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                for action in range(self.number_of_actions):
                    next_row, next_column = self._get_next_state(action)
                    self.state_transitions[row, column, 
                                           action, 
                                           next_row, next_column] = 1

        self._print_environment()

    def _print_environment(self) -> None:
        """Print all the data related to the environment."""

        # np.set_printoptions(threshold=np.inf) # type: ignore

        print(f"Initialised environment of size {self.number_of_rows} {self.number_of_columns}.")
        print(f"Created {self.number_of_states} states.")
        print(f"Every state is equivalent & shares the same actions, for a total of {self.number_of_actions} actions per state.")
        print(f"Agent begins at {self.start_state} & terminates at {self.terminal_states}.")
        print(f"Created {self.number_of_obstacles} obstacles.")
        print(f"Created rewards function.")
        for state in self.rewards:
            print(f"state: {state}  reward: {self.rewards[state]}.")
        print(f"Created equiprobable policy.")
        # for policy in self.policy:
        #     print(policy)
        print(f"Created state transition function.")
        # for state_transition in self.state_transitions:
        #     print(state_transition)

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

    def _get_next_state(self, action: int) -> tuple[int, int]:
        """Get the next state given an action."""

        # Get the current state
        row, column = self.state

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
        match state:
            case (row, column):
                for row_, column_, reward in self.rewards:
                    # Check if reward state
                    if row == row_ and column == column_:
                        return reward
                return 0
            case _: # Invalid state
                raise ValueError(f"Invalid state: {state}")
            
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
    
    def bellman_equation_update_rule(self, ):
        pass
    

    def iterative_policy_evaluation(self, pi: dict, theta: float = 0.0001):
        """Iterative policy evaluation algorithm to find state-value function."""

        # Initialise state-value function values to zero, 
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
                    V[state] = self.bellman_equation_update_rule()


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
