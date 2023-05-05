# RL-Grid-World
Reinforcement Learning Grid World Application.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project is a simple grid world application. It is meant to be a simple, yet accurate, implementation of reinforcement learning. It is meant to be a learning experience for myself, before I move onto approximation methods. 

## Installation
This project is written in Python 3.11.3. It uses the following libraries:
- numpy
- matplotlib

To install these libraries, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the application, run the following command:
```bash
python main.py
```

## Examples
The following are examples of the application running:
![Grid world with initial value function & equiprobable policy](https://github.com/ctorrington/RL-Grid-World/blob/main/images/example0.png?raw=true)

This image shows the initial value function. The value function is initialised to 0. The rewards are initialised to 1 for the terminal states & 0 for all other states. The stochastic policy is initialised to equiprobable actions for each state. The agent is initialised randomly within the GridWorld. The darker squares are obstacles. The agent cannot move into these squares.

![Grid world with optimal value function & optimal policy](https://github.com/ctorrington/RL-Grid-World/blob/main/images/example1.png?raw=true)

This images shows the optimal value function & optimal policy. The optimal value function is calculated using the Bellman optimality equation. The optimal policy is calculated using the optimal value function. The optimal policy is deterministic.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
