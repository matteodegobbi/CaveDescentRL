import collections
import time
from typing_extensions import Self

import pygame
import numpy as np
import torch

import wrappers
from rocket_environment import RocketEnvironment, SCREEN_WIDTH, SCREEN_HEIGHT, Action


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
    env = RocketEnvironment(graphics_on=render)

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
            action = np.argmax(q_values) if np.random.random() > epsilon else np.random.choice([Action.PRESSED, Action.RELEASED])

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
    env = RocketEnvironment(graphics_on=True)

    running = True

    clock = pygame.time.Clock()

    # Construct the network
    network_path = "saves/save_1746316784_700.pt"
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

                if event.type == pygame.KEYDOWN:
                    print(event.key)
                    if event.key == pygame.K_x:
                        env.are_lasers_drawn = not env.are_lasers_drawn

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
    env = RocketEnvironment(graphics_on=True)

    running = True
    clock = pygame.time.Clock()

    while running:
        done = False
        env.reset()
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        env.are_lasers_drawn = not env.are_lasers_drawn

            keys = pygame.key.get_pressed()
            action = Action.PRESSED if keys[pygame.K_SPACE] else Action.RELEASED

            next_state, reward, terminated,truncated = env.step(action)
            done = terminated

            if env.graphics_on:
                env.draw(screen)
                clock.tick(60)

            if done:
                if env.graphics_on:
                    pygame.time.delay(500)
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