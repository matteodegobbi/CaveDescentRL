import pygame
import numpy as np
from enum import IntEnum
import random

PLAYER_X = 300
PLAYER_Y = 300
class Action(IntEnum):
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

        # Sudden jump with 1/100 probability
        if random.randint(1, 100) < 5:
            jump_offset = random.choice([-3, 3])
            self.vel += jump_offset
            # Clamp after sudden jump
            if self.rect.top < 0:
                self.rect.top = 0
            if self.rect.bottom > 600:
                self.rect.bottom = 600

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

def discretize_state(state):
    # Expect state[0] ∈ [0.0, 1.0], state[1] ∈ ~[-1.5, 1.5]
    y_bins = 50
    v_bins = 20
    y = min(int(state[0] * y_bins), y_bins - 1)
    v = min(int((state[1] + 1.5) / 3.0 * v_bins), v_bins - 1)  # Shift and scale to [0,1]
    return y * v_bins + v
graphics_on =  True
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
player = Player(PLAYER_X, PLAYER_Y)
obstacles = [Obstacle(0,0,800,200),Obstacle(0,400,800,200)]  # Add obstacles later if needed
env = Environment(player, obstacles)

running = True
state = env.reset()

clock = pygame.time.Clock()
Q = np.zeros((1000,2))

epsilon = 0.1
alpha = 0.1
gamma = 0.5
iters = 0

rewards = []
episodes_count = 0

while running:
    done = False
    state = env.reset()
    reward_sum = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        '''
        keys = pygame.key.get_pressed()
        action = Action.PRESSED if keys[pygame.K_SPACE] else Action.RELEASED
        '''
        if np.random.rand() < epsilon:
            action = np.random.choice([Action.PRESSED, Action.RELEASED])  # Explore
        else:
            action = np.argmax(Q[discretize_state(state)])  # Exploit

        next_state, reward, done, _ = env.step(action)


        Q[discretize_state(state), action] += alpha * (reward + gamma * np.max(Q[discretize_state(next_state)]) - Q[discretize_state(state), action])

        reward_sum += reward
        #print(state, reward, done)
        if graphics_on:
            env.draw(screen)
            clock.tick(60)

        if done:
            if graphics_on:
                pygame.time.delay(500)
        state = next_state
    rewards.append(reward_sum)
    episodes_count += 1
    graphics_on = (episodes_count % 20 == 0)
    if episodes_count % 10 == 0:
        mean_return = np.mean(rewards[-100:])
        std_return = np.std(rewards[-100:])
        recent_returns = rewards[-10:]
        returns_str = " ".join(map(str, recent_returns))
        print(
            f"Episode {episodes_count}, mean 100-episode return {mean_return:.2f} +-{std_return:.2f}, returns {returns_str}")

env.close()
