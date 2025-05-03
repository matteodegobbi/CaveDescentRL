import collections
import time
from typing_extensions import Self

import pygame
import numpy as np
from enum import IntEnum, Enum
import random

import torch
from pygame import Vector2
from perlin_numpy import generate_perlin_noise_2d
import math

import wrappers

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
    def ray_segment_intersection(self,point, angle_deg, p_seg1, p_seg2):
        px, py = point
        x1, y1 = p_seg1
        x2, y2 = p_seg2

        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        rx, ry = dx, dy
        sx, sy = x2 - x1, y2 - y1
        denom = rx * sy - ry * sx
        if abs(denom) < 1e-10:
            return None  # Parallel, no intersection

        # Solve for t and u
        t_num = (x1 - px) * sy - (y1 - py) * sx
        u_num = (x1 - px) * ry - (y1 - py) * rx
        t = t_num / denom
        u = u_num / denom

        if t >= 0 and 0 <= u <= 1:
            intersection_x = px + t * rx
            intersection_y = py + t * ry
            distance = math.hypot(intersection_x - px, intersection_y - py)
            return distance
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

            curr_dist_right = self.ray_segment_intersection(self.player.rect.center,0, obstacle.point1,
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
        # clamp to avoide infinity
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

        return state

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
            # obstacle_gap = np.interp(i, [0, n], [self.get_obstacle_size(self.total_level - 1), self.get_obstacle_size(self.total_level)])
            obstacle_gap = 250
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
    # def get_obstacle_size(self, level):
    #     return max(-10 * level + 300, 150)

    def move_obstacles(self, speed = 5):
        for obstacle in self.obstacles:
            obstacle.point1.x -= speed
            obstacle.point2.x -= speed
class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, learning_rate = 0.1) -> None:
        input_size = 5
        hidden_layer_size = 50
        output_size = 2
        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, output_size),
        ).to(self.device)

        # Define an optimizer (most likely from `torch.optim`).
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        # Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()

        # PyTorch uses uniform initializer $U[-1/sqrt n, 1/sqrt n]$ for both weights and biases.
        # Keras uses Glorot (also known as Xavier) uniform for weights and zeros for biases.
        # In some experiments, the Keras initialization works slightly better for RL,
        # so we use it instead of the PyTorch initialization; but feel free to experiment.
        self._model.apply(wrappers.torch_init_with_xavier_and_zeros)

        self.initial_time = round(time.time())

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: Self) -> None:
        self._model.load_state_dict(other._model.state_dict())

    def load(self, network_path):
        self._model.load_state_dict(torch.load(network_path, weights_only=True))
        self._model.eval()

    def save(self, episode):
        torch.save(self._model.state_dict(),f"./saves/save_{self.initial_time}_{episode}.pt")

    def count_trainable_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

def train_rocket(render = False):
    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    env = Environment(graphics_on=render)

    running = True

    epsilon_start = 1
    epsilon_final_at = 2000
    epsilon_final = 0.1
    epsilon = epsilon_start
    alpha = 0.0001
    gamma = 0.99
    batch_size = 32
    target_update_freq = 300
    save_every = 100

    clock = pygame.time.Clock()

    # Construct the network
    network = Network(learning_rate=alpha)
    network_hat = Network(learning_rate=alpha)

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = wrappers.ReplayBuffer(max_length=50000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    step = 0
    episode = 0
    rewards = []
    while running:
        # Perform episode
        state, done = env.reset(), False
        reward_sum = 0
        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            # Choose an action.
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values) #if np.random.random() > epsilon else np.random.choice([Action.PRESSED, Action.RELEASED])

            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            reward_sum += reward


            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # If the `replay_buffer` is large enough, perform training using
            # a batch of `args.batch_size` uniformly randomly chosen transitions.

            if len(replay_buffer) < batch_size:
                continue

            # The `replay_buffer` offers a method with signature
            #   sample(self, size, generator=np.random, replace=True) -> list[Transition]
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is much faster, and hence the default) or without.
            # By default, `np.random` is used to generate the random indices, but you can
            # pass your own `np.random.RandomState` instance.

            batch = replay_buffer.sample(batch_size)

            batch_states = np.array([t.state for t in batch])
            q_vals = network.predict(batch_states)
            batch_next_states = np.array([t.next_state for t in batch])
            q_next_vals = network_hat.predict(batch_next_states)

            batch_actions = [t.action for t in batch]
            batch_rewards = [t.reward for t in batch]
            batch_dones = [t.done for t in batch]
            q_vals[np.arange(batch_size), batch_actions] = np.array(batch_rewards) + gamma * np.max(
                q_next_vals, axis=1) * (1 - np.array(batch_dones))

            # After you compute suitable targets, you can train the network by network.train
            network.train(batch_states, q_vals)
            state = next_state

            step += 1
            if step % target_update_freq == 0:
                step = 0
                network_hat.copy_weights_from(network)

            if env.graphics_on:
                env.draw(screen)
                clock.tick(60)
            if done:
                if env.graphics_on:
                    pygame.time.delay(500)
        rewards.append(reward_sum)
        episode += 1
        if epsilon_final_at:
            epsilon = np.interp(episode + 1, [0, epsilon_final_at], [epsilon, epsilon_final])
        if episode % save_every == 0:
            network.save(episode)
        if episode % 10 == 0:
            mean_return = np.mean(rewards[-100:])
            std_return = np.std(rewards[-100:])
            recent_returns = rewards[-10:]
            returns_str = " ".join(map(str, recent_returns))
            print(
                f"Episode {episode}, mean 100-episode return {mean_return:.2f} +-{std_return:.2f}, returns {returns_str}")




def run_rocket():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    env = Environment(graphics_on=True)

    running = True

    clock = pygame.time.Clock()

    # Construct the network
    network_path = "saves/save_1746313177_500.pt"
    network = Network()
    network.load(network_path)
    network_hat = Network()
    network_hat.load(network_path)

    while running:
        # Perform episode
        state, done = env.reset(), False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Choose an action.
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)

            state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated

            if env.graphics_on:
                env.draw(screen)
                clock.tick(60)
            if done:
                if env.graphics_on:
                    pygame.time.delay(500)

def human_play_rocket():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    env = Environment(graphics_on=True)

    running = True
    state = env.reset()
    clock = pygame.time.Clock()


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

            keys = pygame.key.get_pressed()
            action = Action.PRESSED if keys[pygame.K_SPACE] else Action.RELEASED

            next_state, reward, terminated,truncated = env.step(action)
            done = terminated
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
        if episodes_count % 10 == 0:
            mean_return = np.mean(rewards[-100:])
            std_return = np.std(rewards[-100:])
            recent_returns = rewards[-10:]
            returns_str = " ".join(map(str, recent_returns))
            print(
                f"Episode {episodes_count}, mean 100-episode return {mean_return:.2f} +-{std_return:.2f}, returns {returns_str}")
        if episodes_count == 10000:
            break
    env.close()

import  argparse
def main():
    parser = argparse.ArgumentParser(description="Rocket program controller")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--human', action='store_true', help='Play as a human')
    group.add_argument('--run', action='store_true', help='Run the rocket automatically')
    group.add_argument('--train', action='store_true', help='Train the rocket')

    args = parser.parse_args()

    if args.human:
        human_play_rocket()
    elif args.run:
        run_rocket()
    elif args.train:
        train_rocket()

if __name__ == '__main__':
    main()