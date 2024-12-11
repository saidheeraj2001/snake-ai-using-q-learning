
# Snake AI Project: TEAM DD

## Overview
This project is an AI-controlled Snake game that uses Reinforcement Learning (RL) techniques—specifically Q-learning via a Deep Neural Network—to train an agent to play the classic Snake game.

The core idea is to let the agent learn optimal moves by interacting with the game environment. The game is represented by the `SnakeGameAI` environment, and the agent leverages a Q-network (`Linear_QNet`) and an experience replay memory to gradually improve its decision-making abilities.

## Key Features
- **Automatic Gameplay:** The agent plays the Snake game without human input, continually learning from past experiences.
- **Neural Network-based:** Uses a simple feedforward neural network to approximate Q-values for state-action pairs.
- **Experience Replay:** Stores past experiences `(state, action, reward, next_state, done)` and samples from them to stabilize learning.
- **Performance Metrics:** Plots scores over time to visualize learning progress.

## Project Structure
- `game.py`: Defines the `SnakeGameAI` environment, including rendering, snake movement, food placement, and collision detection.
- `model.py`: Contains the `Linear_QNet` class (a neural network model) and the `QTrainer` class for training it.
- `agent.py`: Implements the `SnakeAgent` class, which uses Q-learning to make decisions, store experiences, and improve over time.
- `helper.py`: Includes utility functions such as `plot` for graphing the learning progress.
- `snake_agent.py` (or your chosen main script): Coordinates training the agent and integrates all components.

## Installation and Setup
1. **Prerequisites:**
   - **Python 3.6+**
   - **PyTorch:** For the neural network and training (`pip install torch torchvision torchaudio`)
   - **NumPy:** For numerical operations (`pip install numpy`)
   - **Pygame:** For game rendering and event handling (`pip install pygame`)
   - **Matplotlib:** For plotting training progress (`pip install matplotlib`)

2. **Project Files:**
   Clone or download this repository and place all the `.py` files and related assets in the same directory.

3. **Folder Structure:**
   ```
   project/
   ├─ game.py
   ├─ helper.py
   ├─ model.py
   ├─ snake_agent.py

## Usage Instructions
1. **Run the Training Script:**
   ```bash
   python snake_agent.py
   ```
   This will start the training loop. The agent will begin playing Snake, making random moves initially, and then gradually improving as it learns from its experiences.

2. **During Training:**
   - A Pygame window will show the Snake game in progress.
   - The training loop will display in the console:
     - The current game number (`Game n`),
     - The agent's score for that game (`Score: x`),
     - The best record so far (`Record: y`).
   - A plotting window (matplotlib) will show:
     - The scores for each game (blue line),
     - The running average score (orange line).

3. **Stopping and Resuming:**
   - Press the window’s close button to stop training at any time.
   - The best model parameters are saved automatically in `./saved_models/model.pth`.
   - You can resume training by running `snake_agent.py` again. The code can be adapted to load the saved model weights if desired.

4. **Adjusting Parameters:**
   - **Hyperparameters:** You can modify `MEMORY_LIMIT`, `BATCH_SIZE`, `LEARNING_RATE`, and the neural network’s layer sizes in `agent.py` or `model.py`.
   - **Game Settings:** Adjust the speed, screen size, and block size in `game.py`.
   - **Exploration Rate Decay:** Controlled by `epsilon` logic in `snake_agent.py` (or wherever the agent chooses actions). Tweak it to balance exploration and exploitation.

## Tips for Better Results
- **Longer Training:** Run the training for many games. RL solutions often require thousands of episodes before performance improves significantly.
- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and network architectures.
- **Reward Function:** Consider adjusting the reward structure in `game.py` to encourage desired behaviors (like reaching the food more reliably).
- **Environment Complexity:** Start simple and gradually increase complexity (e.g., larger grid, faster speeds) after the agent has learned the basics.

## Contributing
- **Bug Reports and Features:** If you improve or modify the code, consider sharing back contributions to help other students and researchers.
- **Documentation:** Extend or refine this documentation to assist future users.

## Contact and Support
For questions, suggestions, or issues, you can reach out on the appropriate channels where you obtained this project. Any feedback is appreciated to make this project more robust and user-friendly.

---
