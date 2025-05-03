import pygame
import numpy as np
from enum import IntEnum, Enum
from pygame import Vector2
from perlin_numpy import generate_perlin_noise_2d


SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700

OBSTACLE_WIDTH = 80
OBSTACLE_GAP = 250

class Action(IntEnum):
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
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 40, 20)
        self.vel = 0
        self.gravity = 0.3
        self.thrust = -0.8

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def update(self, action: Action):
        if action == Action.PRESSED:
            self.vel += self.thrust
        self.vel += self.gravity
        self.rect.y += int(self.vel)

        # Clamp to screen
        if self.rect.top < 0:
            self.rect.top = 0
            self.vel = 0
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
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
            points = [self.point1, self.point2, Vector2(self.point2.x, SCREEN_HEIGHT),
                      Vector2(self.point1.x, SCREEN_HEIGHT)]
            pygame.draw.polygon(screen, (255, 0, 0), points)

    def collides_with_rect(self, rect):
        return rect.clipline(self.point1, self.point2)


class Environment:
    def __init__(self, graphics_on=True, steps_before_truncation=4000):
        self.background_color = (30, 30, 30)
        self.player = None
        self.obstacles = []

        self.terminated = False
        self.truncated = False
        self.graphics_on = graphics_on
        self.steps_since_episode = 0
        self.steps_before_truncation = steps_before_truncation

        self.noise = None
        self.noise_size = 1024
        self.total_level = 0
        self.reset()

    def reset(self):
        self.noise = generate_perlin_noise_2d((self.noise_size, self.noise_size), (4, 2), tileable=(True, True))
        self.terminated = False
        self.truncated = False
        self.steps_since_episode = 0
        self.obstacles = []
        self.generate_random_obstacles(keep_middle_clear = True, n=50)
        self.total_level = 0

        self.player = Player()
        self.player.set_position(300, SCREEN_HEIGHT / 2)
        return self.get_state()

    def step(self, action):
        self.player.update(action)
        for obstacle in self.obstacles:
            if obstacle.collides_with_rect(self.player.rect):
                self.terminated = True

        # delete old obstacles
        if self.obstacles[0].point1.x < -OBSTACLE_WIDTH:
            self.obstacles.pop(0)

        # generate new obstacles
        RENDER_OFFSET = 100
        if self.obstacles[-1].point1.x + OBSTACLE_WIDTH - RENDER_OFFSET < SCREEN_WIDTH:
            self.generate_random_obstacles(offset_x = SCREEN_WIDTH + RENDER_OFFSET)

        self.move_obstacles()

        reward = 1 if not self.terminated else -100
        if self.steps_since_episode > self.steps_before_truncation:
            self.truncated = True

        self.steps_since_episode += 1
        return self.get_state(), reward, self.terminated, self.truncated

    def get_state(self):
        # y pos and vel are normalized
        state = np.array([
            self.player.rect.y / SCREEN_HEIGHT,
            self.player.vel / 10.0
        ], dtype=np.float32)
        return self.discretize_state(state)

    def discretize_state(self, state):
        # Expect state[0] in [0.0, 1.0], state[1] in ~[-1.5, 1.5]
        y_bins = 50
        v_bins = 20
        y = min(int(state[0] * y_bins), y_bins - 1)
        v = min(int((state[1] + 1.5) / 3.0 * v_bins), v_bins - 1)  # Shift and scale to [0,1]
        return y * v_bins + v

    def draw(self,screen):
        assert self.graphics_on
        screen.fill(self.background_color)
        self.player.draw(screen)
        for obstacle in self.obstacles:
            obstacle.draw(screen)
        pygame.display.flip()

    def close(self):
        if self.graphics_on:
            pygame.quit()

    def generate_random_obstacles(self, n=20, offset_x = 0, keep_middle_clear = False):
        last_x = None
        last_y = None

        total_x = 0
        if len(self.obstacles) > 0:
            self.total_level += 1
            total_x = self.obstacles[-2].total_x - OBSTACLE_WIDTH # -2 because the last is the bottom one and the second last is the top one
            last_x = self.obstacles[-2].point2.x
            last_y = self.obstacles[-2].point2.y

        for i in range(n):
            offset_obstacles_y = np.interp(i, [0, n], [150, 5]) if keep_middle_clear else 5
            obstacle_gap = np.interp(i, [0, n], [self.get_obstacle_size(self.total_level - 1), self.get_obstacle_size(self.total_level)])
            total_x = (total_x + OBSTACLE_WIDTH) % self.noise_size
            x = offset_x + i * OBSTACLE_WIDTH
            y = np.interp(self.noise[total_x, total_x], [-1, 1], [offset_obstacles_y, SCREEN_HEIGHT - offset_obstacles_y - obstacle_gap])
            if last_x is not None and last_y is not None:
                obstacle = Obstacle(Vector2(last_x, last_y), Vector2(x, y), ObstacleType.TOP, total_x)
                self.obstacles.append(obstacle)
                obstacle = Obstacle(Vector2(last_x, last_y + obstacle_gap), Vector2(x, y + obstacle_gap), ObstacleType.BOTTOM,
                                    total_x)
                self.obstacles.append(obstacle)
            last_x = x
            last_y = y
    def get_obstacle_size(self, level):
        return max(-10 * level + 300, 150)

    def move_obstacles(self, speed = 5):
        for obstacle in self.obstacles:
            obstacle.point1.x -= speed
            obstacle.point2.x -= speed

def train_rocket():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    env = Environment(graphics_on=not False)

    running = True
    state = env.reset()

    clock = pygame.time.Clock()
    Q = np.zeros((1000, 2))

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

            # keys = pygame.key.get_pressed()
            # action = Action.PRESSED if keys[pygame.K_SPACE] else Action.RELEASED

            if np.random.rand() < epsilon:
                action = np.random.choice([Action.PRESSED, Action.RELEASED])  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, terminated,truncated = env.step(action)
            done = terminated or truncated
            if truncated:
                print("TRUNCATED ", env.steps_since_episode)

            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            reward_sum += reward

            if env.graphics_on:
                env.draw(screen)
                clock.tick(60)

            if done:
                if env.graphics_on:
                    pygame.time.delay(500)
            state = next_state
        rewards.append(reward_sum)
        episodes_count += 1
        # env.graphics_on = (episodes_count % 100 == 0)
        if episodes_count % 10 == 0:
            mean_return = np.mean(rewards[-100:])
            std_return = np.std(rewards[-100:])
            recent_returns = rewards[-10:]
            returns_str = " ".join(map(str, recent_returns))
            print(
                f"Episode {episodes_count}, mean 100-episode return {mean_return:.2f} +-{std_return:.2f}, returns {returns_str}")

    env.close()


import sys
def main():
    train_rocket()


if __name__ == '__main__':
    main()