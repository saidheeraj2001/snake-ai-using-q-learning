import pygame
import random
import time

pygame.init()

WIDTH, HEIGHT = 600, 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FOOD_COLOR = (200, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()
SNAKE_SPEED = 15

snake_pos = [[100, 50], [90, 50], [80, 50]]
snake_direction = "RIGHT"
food_pos = [random.randrange(1, (WIDTH // 10)) * 10, random.randrange(1, (HEIGHT // 10)) * 10]
food_spawn = True
score = 0

def random_color():
    return random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)

snake_color = random_color()

def game_over():
    font = pygame.font.SysFont("comicsans", 50)
    game_over_surface = font.render(f"Game Over! Score: {score}", True, FOOD_COLOR)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (WIDTH / 2, HEIGHT / 4)
    screen.fill(BLACK)
    screen.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    quit()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and not snake_direction == "DOWN":
        snake_direction = "UP"
    if keys[pygame.K_DOWN] and not snake_direction == "UP":
        snake_direction = "DOWN"
    if keys[pygame.K_LEFT] and not snake_direction == "RIGHT":
        snake_direction = "LEFT"
    if keys[pygame.K_RIGHT] and not snake_direction == "LEFT":
        snake_direction = "RIGHT"

    if snake_direction == "UP":
        snake_pos[0][1] -= 10
    if snake_direction == "DOWN":
        snake_pos[0][1] += 10
    if snake_direction == "LEFT":
        snake_pos[0][0] -= 10
    if snake_direction == "RIGHT":
        snake_pos[0][0] += 10

    if snake_pos[0][0] < 0 or snake_pos[0][0] >= WIDTH or snake_pos[0][1] < 0 or snake_pos[0][1] >= HEIGHT:
        game_over()

    for block in snake_pos[1:]:
        if snake_pos[0] == block:
            game_over()

    snake_pos.insert(0, list(snake_pos[0]))
    if snake_pos[0] == food_pos:
        score += 1
        snake_color = random_color()
        food_spawn = False
    else:
        snake_pos.pop()

    if not food_spawn:
        while True:
            food_pos = [random.randrange(1, (WIDTH // 10)) * 10, random.randrange(1, (HEIGHT // 10)) * 10]
            if food_pos not in snake_pos:
                break
    food_spawn = True

    screen.fill(BLACK)

    for block in snake_pos:
        pygame.draw.rect(screen, snake_color, pygame.Rect(block[0], block[1], 10, 10))

    pygame.draw.rect(screen, FOOD_COLOR, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

    font = pygame.font.SysFont("comicsans", 20)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, [10, 10])

    pygame.display.update()
    clock.tick(SNAKE_SPEED)
