import pygame
import numpy as np
from enum import Enum
PLAYER_X = 300
PLAYER_Y = 400
class Action(Enum):
    RELEASED = 0
    PRESSED = 1
class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class Player:
    def __init__(self, x, y):
        self.rect =pygame.Rect(x, y, 40, 20)
        self.vel = 0
        self.gravity = 0.3
        self.thrust = -0.8

    def update(self, action: Action):
        if action == Action.PRESSED:
            self.vel += self.thrust
        self.vel += self.gravity
        self.rect.y += int(self.vel)

        # Clamp to screen
        if self.rect.top < 0:
            self.rect.top = 0
            self.vel = 0
        if self.rect.bottom > 600:
            self.rect.bottom = 600
            self.vel = 0

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), self.rect)  # green rocket


class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)  # red obstacle


# --- Environment class ---
class Environment:
    def __init__(self, player, obstacles):
        self.player = player
        self.obstacles = obstacles
        self.background_color = (30, 30, 30)
        self.done = False

    def reset(self):
        self.player = Player(PLAYER_X,PLAYER_Y)
        self.done = False
        return self.get_state()

    def step(self, action):
        self.player.update(action)
        for obstacle in self.obstacles:
            if self.player.rect.colliderect(obstacle.rect):
                self.done = True

        reward = 1 if not self.done else -100
        return self.get_state(), reward, self.done, {}

    def get_state(self):
        return np.array([
            self.player.rect.y / 600,
            self.player.vel / 10.0
        ], dtype=np.float32)

    def draw(self,screen):
        assert graphics_on
        screen.fill(self.background_color)
        self.player.draw(screen)
        for obstacle in self.obstacles:
            obstacle.draw(screen)
        pygame.display.flip()

    def close(self):
        if graphics_on:
            pygame.quit()


graphics_on = True
pygame.init()
screen = pygame.display.set_mode((800, 600))
player = Player(PLAYER_X, PLAYER_Y)
obstacles = [Obstacle(0,0,800,100),Obstacle(0,500,800,100)]  # Add obstacles later if needed
env = Environment(player, obstacles)

running = True
state = env.reset()

clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = Action.PRESSED if keys[pygame.K_SPACE] else Action.RELEASED

    state, reward, done, _ = env.step(action)
    print(state, reward, done)
    if graphics_on:
        env.draw(screen)
        clock.tick(60)

    if done:
        if graphics_on:
            pygame.time.delay(500)
        state = env.reset()

env.close()
