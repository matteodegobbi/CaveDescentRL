


import pygame
import numpy as np

# --- Player class ---
class Player:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 40, 20)
        self.vel = 0
        self.gravity = 0.5
        self.thrust = -10

    def update(self, action):
        if action == 1:
            self.vel = self.thrust
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


# --- Obstacle class ---
class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)  # red obstacle


# --- Environment class ---
class Environment:
    def __init__(self, screen, player, obstacles):
        self.screen = screen
        self.player = player
        self.obstacles = obstacles
        self.background_color = (30, 30, 30)
        self.clock = pygame.time.Clock()
        self.done = False

    def reset(self):
        self.player = Player(100, 300)
        self.done = False
        return self.get_state()

    def step(self, action):
        self.player.update(action)

        # Collision detection
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

    def draw(self):
        self.screen.fill(self.background_color)
        self.player.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        pygame.display.flip()

    def close(self):
        pygame.quit()


# --- Main loop ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
player = Player(100, 300)
obstacles = []  # Add obstacles later if needed
env = Environment(screen, player, obstacles)

running = True
state = env.reset()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = 1 if keys[pygame.K_SPACE] else 0

    state, reward, done, _ = env.step(action)
    env.draw()
    env.clock.tick(60)

    if done:
        pygame.time.delay(500)
        state = env.reset()

env.close()
