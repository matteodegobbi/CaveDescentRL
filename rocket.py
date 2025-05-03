from ssl import get_server_certificate

import pygame
import numpy as np
from enum import IntEnum, Enum
import random
from pygame import Vector2
from perlin_numpy import generate_perlin_noise_2d


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

PLAYER_X = 300
PLAYER_Y = SCREEN_HEIGHT / 2

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
    def __init__(self, player, graphics_on=True, steps_before_truncation=4000):
        self.player = player
        self.background_color = (30, 30, 30)
        self.obstacles = []

        self.terminated = False
        self.truncated = False
        self.graphics_on = graphics_on
        self.steps_since_episode = 0
        self.steps_before_truncation = steps_before_truncation

        self.noise = None
        self.noise_size = 1024
        self.reset()

    def reset(self):
        self.noise = generate_perlin_noise_2d((self.noise_size, self.noise_size), (4, 2), tileable=(True, True))
        self.player = Player(PLAYER_X,PLAYER_Y)
        self.terminated = False
        self.truncated = False
        self.steps_since_episode = 0
        self.obstacles = []
        self.generate_random_obstacles()
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

    def vertical_ray_segment_intersection_down(self,point, p_seg1, p_seg2):
        px,py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2
        # Check if segment crosses vertical line x = px
        if (x1 - px) * (x2 - px) > 0:
            return None  # Both endpoints on same side of vertical line

        # Handle vertical segment
        if x1 == x2:
            if x1 != px:
                return None
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            if py <= y_bottom:
                return max(py, y_top) - py
            return None

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_intersect = m * px + b

        if min(y1, y2) <= y_intersect <= max(y1, y2) and y_intersect >= py:
            return y_intersect - py
        return None

    def vertical_ray_segment_intersection_up(self,point, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2
        if (x1 - px) * (x2 - px) > 0:
            return None

        if x1 == x2:
            if x1 != px:
                return None
            y_top = min(y1, y2)
            y_bottom = max(y1, y2)
            if py >= y_top:
                return py - min(py, y_bottom)
            return None

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y_intersect = m * px + b

        if min(y1, y2) <= y_intersect <= max(y1, y2) and y_intersect <= py:
            return py - y_intersect
        return None

    def horizontal_ray_segment_intersection_right(self,point, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2
        if (y1 - py) * (y2 - py) > 0:
            return None
        if y1 == y2:
            if y1 != py:
                return None
            x_left = min(x1, x2)
            x_right = max(x1, x2)
            if px <= x_right:
                return max(px, x_left) - px
            return None
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        if m == 0:
            return None
        x_intersect = (py - b) / m
        if min(x1, x2) <= x_intersect <= max(x1, x2) and x_intersect >= px:
            return x_intersect - px
        return None

    def get_obstacle_distances(self):
        min_dist_down = float('inf');
        min_dist_up = float('inf');
        min_dist_right = float('inf');
        for obstacle in self.obstacles:
            curr_dist_down = self.vertical_ray_segment_intersection_down(self.player.rect.center, obstacle.point1,
                                                                         obstacle.point2)
            curr_dist_up = self.vertical_ray_segment_intersection_up(self.player.rect.center, obstacle.point1,
                                                                     obstacle.point2)

            curr_dist_right = self.horizontal_ray_segment_intersection_right(self.player.rect.center, obstacle.point1,
                                                                     obstacle.point2)
            if curr_dist_down != None and curr_dist_down < min_dist_down:
                min_dist_down = curr_dist_down

            if curr_dist_up != None and curr_dist_up < min_dist_up:
                min_dist_up = curr_dist_up

            if curr_dist_right != None and curr_dist_right< min_dist_right:
                min_dist_right= curr_dist_right
        return min_dist_down, min_dist_up, min_dist_right


    def get_state(self):
        # y pos and vel are normalized
        dist_obst_down, dist_obst_up,dist_obst_right = self.get_obstacle_distances()
        dist_obst_down = min(dist_obst_down, SCREEN_HEIGHT)
        dist_obst_up = min(dist_obst_up , SCREEN_HEIGHT)
        dist_obst_right = min(dist_obst_right, SCREEN_WIDTH)

        state = np.array([
            self.player.rect.y / SCREEN_HEIGHT,
            self.player.vel / 10.0,
            dist_obst_down / SCREEN_HEIGHT,
            dist_obst_up / SCREEN_HEIGHT,
            dist_obst_right / SCREEN_WIDTH,
        ], dtype=np.float32)

        return self.discretize_state(state)


    def discretize_state(self, state):
        # state: [y, velocity, dist_down, dist_up, dist_right]
        y_bins = 10
        v_bins = 10
        d_bins = 10

        y = min(int(state[0] * y_bins), y_bins - 1)
        v = min(int((state[1] + 1.5) / 3.0 * v_bins), v_bins - 1)
        d_down = min(int(state[2] * d_bins), d_bins - 1)
        d_up = min(int(state[3] * d_bins), d_bins - 1)
        d_right = min(int(state[4] * d_bins), d_bins - 1)

        num_states = y_bins * v_bins * d_bins * d_bins * d_bins
        num_actions = 2
        #print("Q-table size:", num_states * num_actions)
        return ((((y * v_bins + v) * d_bins + d_down) * d_bins + d_up) * d_bins + d_right)


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

def train_rocket():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    player = Player(PLAYER_X, PLAYER_Y)
    env = Environment(player,graphics_on= False)

    running = True
    state = env.reset()

    clock = pygame.time.Clock()
    Q = np.zeros(( 200000, 2))

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
        env.graphics_on = (episodes_count % 100 == 0 or episodes_count > 500)
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