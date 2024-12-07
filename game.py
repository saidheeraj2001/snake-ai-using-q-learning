
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class MovementDirection(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Coordinate = namedtuple('Coordinate', 'x, y')

COLOR_WHITE = (255, 255, 255)
COLOR_RED = (200, 0, 0)
COLOR_BLUE_PRIMARY = (0, 0, 255)
COLOR_BLUE_SECONDARY = (0, 100, 255)
COLOR_BLACK = (0, 0, 0)

CELL_SIZE = 20
GAME_SPEED = 40

class SnakeGameAI:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('TEAM DD SNAKE')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = MovementDirection.RIGHT
        self.head = Coordinate(self.width / 2, self.height / 2)
        self.snake_body = [
            self.head,
            Coordinate(self.head.x - CELL_SIZE, self.head.y),
            Coordinate(self.head.x - (2 * CELL_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self._spawn_food()
        self.frame_count = 0

    def _spawn_food(self):
        while True:
            x = random.randint(0, (self.width - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            y = random.randint(0, (self.height - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
            self.food = Coordinate(x, y)
            if self.food not in self.snake_body:
                break

    def play_turn(self, move):
        self.frame_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._update_snake_position(move)
        self.snake_body.insert(0, self.head)
        reward = 0
        game_over = False
        if self.check_collision() or self.frame_count > 100 * len(self.snake_body):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._spawn_food()
        else:
            self.snake_body.pop()
        self._render_game()
        self.clock.tick(GAME_SPEED)
        return reward, game_over, self.score

    def check_collision(self, point=None):
        if point is None:
            point = self.head
        if point.x > self.width - CELL_SIZE or point.x < 0 or point.y > self.height - CELL_SIZE or point.y < 0:
            return True
        if point in self.snake_body[1:]:
            return True
        return False

    def _render_game(self):
        self.display.fill(COLOR_BLACK)
        for body_part in self.snake_body:
            pygame.draw.rect(self.display, COLOR_BLUE_PRIMARY, pygame.Rect(body_part.x, body_part.y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(self.display, COLOR_BLUE_SECONDARY, pygame.Rect(body_part.x + 4, body_part.y + 4, 12, 12))
        pygame.draw.rect(self.display, COLOR_RED, pygame.Rect(self.food.x, self.food.y, CELL_SIZE, CELL_SIZE))
        score_text = font.render("Score: " + str(self.score), True, COLOR_WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()

    def _update_snake_position(self, move):
        directions = [MovementDirection.RIGHT, MovementDirection.DOWN, MovementDirection.LEFT, MovementDirection.UP]
        current_index = directions.index(self.direction)
        if np.array_equal(move, [1, 0, 0]):
            new_direction = directions[current_index]
        elif np.array_equal(move, [0, 1, 0]):
            new_direction = directions[(current_index + 1) % 4]
        else:
            new_direction = directions[(current_index - 1) % 4]
        self.direction = new_direction
        x = self.head.x
        y = self.head.y
        if self.direction == MovementDirection.RIGHT:
            x += CELL_SIZE
        elif self.direction == MovementDirection.LEFT:
            x -= CELL_SIZE
        elif self.direction == MovementDirection.DOWN:
            y += CELL_SIZE
        elif self.direction == MovementDirection.UP:
            y -= CELL_SIZE
        self.head = Coordinate(x, y)
