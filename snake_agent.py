
import torch
import random
import numpy as np
from collections import deque
import pygame
from game import SnakeGameAI, MovementDirection, Coordinate
from model import NeuralNet, QLearningTrainer
from helper import plot_progress

MEMORY_CAPACITY = 100_000
MINIBATCH_SIZE = 1000
LEARNING_RATE = 0.001

class SnakeAgent:
    def __init__(self):
        self.game_count = 0
        self.epsilon = 0  # Exploration factor
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.model = NeuralNet(11, 256, 3)
        self.trainer = QLearningTrainer(self.model, learning_rate=LEARNING_RATE, discount_factor=self.gamma)

    def evaluate_state(self, game_instance):
        head = game_instance.snake_body[0]
        left = Coordinate(head.x - 20, head.y)
        right = Coordinate(head.x + 20, head.y)
        up = Coordinate(head.x, head.y - 20)
        down = Coordinate(head.x, head.y + 20)

        direction_left = game_instance.direction == MovementDirection.LEFT
        direction_right = game_instance.direction == MovementDirection.RIGHT
        direction_up = game_instance.direction == MovementDirection.UP
        direction_down = game_instance.direction == MovementDirection.DOWN

        state = [
            # Danger straight
            (direction_right and game_instance.check_collision(right)) or
            (direction_left and game_instance.check_collision(left)) or
            (direction_up and game_instance.check_collision(up)) or
            (direction_down and game_instance.check_collision(down)),

            # Danger right
            (direction_up and game_instance.check_collision(right)) or
            (direction_down and game_instance.check_collision(left)) or
            (direction_left and game_instance.check_collision(up)) or
            (direction_right and game_instance.check_collision(down)),

            # Danger left
            (direction_down and game_instance.check_collision(right)) or
            (direction_up and game_instance.check_collision(left)) or
            (direction_right and game_instance.check_collision(up)) or
            (direction_left and game_instance.check_collision(down)),

            # Movement direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food location relative to snake
            game_instance.food.x < game_instance.head.x,
            game_instance.food.x > game_instance.head.x,
            game_instance.food.y < game_instance.head.y,
            game_instance.food.y > game_instance.head.y
        ]

        return np.array(state, dtype=int)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_from_memory(self):
        if len(self.memory) > MINIBATCH_SIZE:
            minibatch = random.sample(self.memory, MINIBATCH_SIZE)
        else:
            minibatch = self.memory

        states, actions, rewards, next_states, dones = zip(*minibatch)
        self.trainer.perform_training_step(states, actions, rewards, next_states, dones)

    def train_step(self, state, action, reward, next_state, done):
        self.trainer.perform_training_step(state, action, reward, next_state, done)

    def decide_action(self, state):
        self.epsilon = 80 - self.game_count
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train_agent():
    score_history = []
    mean_score_history = []
    cumulative_score = 0
    high_score = 0
    agent = SnakeAgent()
    game = SnakeGameAI()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                plt.close()
                pygame.quit()
                quit()

        current_state = agent.evaluate_state(game)
        chosen_move = agent.decide_action(current_state)

        reward, game_over, score = game.play_turn(chosen_move)
        next_state = agent.evaluate_state(game)

        agent.train_step(current_state, chosen_move, reward, next_state, game_over)
        agent.memorize(current_state, chosen_move, reward, next_state, game_over)

        if game_over:
            game.reset()
            agent.game_count += 1
            agent.train_from_memory()

            if score > high_score:
                high_score = score
                agent.model.save_model()

            print(f'Game {agent.game_count}, Score: {score}, High Score: {high_score}')

            score_history.append(score)
            cumulative_score += score
            mean_score = cumulative_score / agent.game_count
            mean_score_history.append(mean_score)
            plot_progress(score_history, mean_score_history)

if __name__ == "__main__":
    train_agent()
