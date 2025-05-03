import pygame
import numpy as np
from enum import Enum

from pygame import Vector2


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

PLAYER_X = 300
PLAYER_Y = SCREEN_HEIGHT / 2

OBSTACLE_WIDTH = 80
OBSTACLE_GAP = 250

from perlin_numpy import generate_perlin_noise_2d


class Action(Enum):
    RELEASED = 0
    PRESSED = 1

class ObstacleType(Enum):
    TOP = 0
    BOTTOM = 1

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
    def __init__(self, point1: Vector2, point2: Vector2, type: ObstacleType, total_x):
        self.point1 = point1
        self.point2 = point2
        self.type = type
        self.total_x = total_x


    def draw(self, screen):
        if self.type == ObstacleType.TOP:
            points = [self.point1, self.point2, Vector2(self.point2.x, 0), Vector2(self.point1.x, 0)]
            pygame.draw.polygon(screen, (255, 0, 0), points)
        else:
            points = [self.point1, self.point2, Vector2(self.point2.x, SCREEN_HEIGHT), Vector2(self.point1.x, SCREEN_HEIGHT)]
            pygame.draw.polygon(screen, (255, 0, 0), points)

    def collides_with_rect(self, rect):
        return rect.clipline(self.point1, self.point2)


# --- Environment class ---
class Environment:
    def __init__(self, player):
        self.player = player
        self.background_color = (30, 30, 30)
        self.obstacles = []

        self.noise = None
        self.noise_size = 1024
        self.reset()

    def reset(self):
        self.noise = generate_perlin_noise_2d((self.noise_size, self.noise_size), (4, 2), tileable=(True, True))
        self.player = Player(PLAYER_X,PLAYER_Y)
        self.obstacles = []
        self.generate_random_obstacles()
        return self.get_state()

    def step(self, action):
        self.player.update(action)
        done = False
        for obstacle in self.obstacles:
            if obstacle.collides_with_rect(self.player.rect):
                done = True

        # delete old obstacles
        if self.obstacles[0].point1.x < -OBSTACLE_WIDTH:
            self.obstacles.pop(0)

        # generate new obstacles
        RENDER_OFFSET = 100
        if self.obstacles[-1].point1.x + OBSTACLE_WIDTH - RENDER_OFFSET < SCREEN_WIDTH:
            self.generate_random_obstacles(offset_x = SCREEN_WIDTH + RENDER_OFFSET)

        self.move_obstacles()

        reward = 1 if not done else -100
        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.array([
            self.player.rect.y / SCREEN_HEIGHT,
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

    def generate_random_obstacles(self, n=20, offset_x = 0):
        offset_obstacles_y = 0
        last_x = None
        last_y = None

        total_x = 0
        if len(self.obstacles) > 0:
            total_x = self.obstacles[-2].total_x - OBSTACLE_WIDTH # -2 because the last is the bottom one and the second last is the top one
            last_x = self.obstacles[-2].point2.x
            last_y = self.obstacles[-2].point2.y

        for i in range(n):
            total_x = (total_x + OBSTACLE_WIDTH) % self.noise_size
            x = offset_x + i * OBSTACLE_WIDTH
            y = np.interp(self.noise[total_x, total_x], [-1, 1], [offset_obstacles_y, SCREEN_HEIGHT - offset_obstacles_y - OBSTACLE_GAP])
            if last_x is not None and last_y is not None:
                self.add_obstacle(last_x, last_y, x, y, total_x)
            last_x = x
            last_y = y

    def add_obstacle(self, x1, y1, x2, y2, total_x):
        obstacle = Obstacle(Vector2(x1, y1), Vector2(x2, y2), ObstacleType.TOP, total_x)
        self.obstacles.append(obstacle)
        obstacle = Obstacle(Vector2(x1, y1 + OBSTACLE_GAP), Vector2(x2, y2 + OBSTACLE_GAP), ObstacleType.BOTTOM, total_x)
        self.obstacles.append(obstacle)



    def move_obstacles(self, speed = 5):
        for obstacle in self.obstacles:
            obstacle.point1.x -= speed
            obstacle.point2.x -= speed

graphics_on = True
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
player = Player(PLAYER_X, PLAYER_Y)
env = Environment(player)

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
    # print(state, reward, done)
    if graphics_on:
        env.draw(screen)
        clock.tick(60)

    if done:
        if graphics_on:
            pygame.time.delay(500)
        state = env.reset()

env.close()
