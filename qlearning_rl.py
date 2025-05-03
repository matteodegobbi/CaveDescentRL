import numpy as np
import random
import argparse

observation_space = 3 ** 9
action_space = 9

class RandomAgent:
    def get_action(self,env,state):
        return self.get_random_move(env)
        
    def get_random_move(self,env):
        available_coords = list(zip(*np.where(env.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            return None
        return random.choice(available)
        
class HumanAgent:
    def get_action(self,env,state):
        available_coords = list(zip(*np.where(env.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            print("No available move for human")
            return None
        return int(input())

class QLearnAgent:

    def __init__(self,epsilon = 0.1,alpha = 0.5,gamma = 0.5):
        self.Q = np.zeros((observation_space, action_space))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self,env,state):
        if np.random.rand() < self.epsilon:
            action = env.sample_action()
        else:
            available_moves = env.get_valid_moves()
            q_values = self.Q[state]
            q_filtered = np.where(available_moves > 0, q_values, -np.inf)
            action = np.argmax(q_filtered)
        return action

    def learn(self, state, action, next_state, reward):
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def load(self, path):
        self.Q = np.load(path)

    def save(self, path):
        np.save(path, self.Q)


class Environment:
    def __init__(self, agent2, prints_board=False):
        self.board = np.full((3, 3), -1)
        self.current_player = 0
        self.prints_board = prints_board
        self.agent2 = agent2
    
    def step(self, move: int):
        self.make_move(move, self.current_player)

        if self.is_done():
            # Current player won
            return self.get_state(), 1, True, False, {}

        if self.prints_board:
            self.print_board()
        self.current_player = self.get_opposite_player()
        other_action = self.agent2.get_action(self,self.get_state())
        if other_action is None:
            # Draw
            return self.get_state(), 0, True, False, {}
        self.make_move(other_action, self.current_player)

        if self.is_done():
            # The other player won
            return self.get_state(), -1, True, False, {}

        self.current_player = self.get_opposite_player()
        # next_state, reward, terminated, truncated, _
        return self.get_state(), 0, False, False, {}

    def make_move(self, move, player):
        one_hot_move = np.zeros(action_space)
        one_hot_move[move] = 1
        coords = (move // 3, move % 3)
        if self.board[coords] != -1:
            raise Exception("Already occupied")
        self.board[coords] = player


    def get_valid_moves(self):
        available_coords = list(zip(*np.where(self.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        moves = np.zeros(action_space)
        for move in available:
            moves[move] = 1
        return moves

    def reset(self):
        self.board = np.full((3, 3), -1)
        self.current_player = 0
        return self.get_state()

    def get_opposite_player(self):
        return 1 - self.current_player
        
    def is_done(self):
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player):
                return True
            if np.all(self.board[:, i] == self.current_player):
                return True
        if np.all(np.diag(self.board) == self.current_player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == self.current_player):
            return True
        return False

    def get_random_move(self):
        available_coords = list(zip(*np.where(self.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            return None
        return random.choice(available)
    def sample_action(self):
        return self.get_random_move()
        
    def print_board(self):
        for row in self.board:
            print("|".join(self.symbol(cell) for cell in row))
            print("-" * 5)
            
    def symbol(self,val):
        if val == 0:
            return "X"
        elif val == 1:
            return "O"
        else:
            return " "

    def get_state(self):
        # Flatten board and shift values
        flat = ((np.array(self.board).flatten()) + 1).astype(int)  # now 0=empty, 1=x, 2=o
        state = 0
        for i in range(9):
            state += flat[i] * (3 ** i)  # base-3 encoding
        return state
def train_q_learn():

    agent1 = QLearnAgent()
    agent2 = RandomAgent()
    env = Environment(agent2)

    training = True
    
    rewards = []
    episodes_count = 0
    
    while training:
        # Perform episode
        state = env.reset()
    
        done = False
        reward_sum = 0
        while not done:
            action = agent1.get_action(env, state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the action-value estimates
            agent1.learn(state, action, next_state, reward)
    
            state = next_state
            reward_sum += reward
    
        rewards.append(reward_sum)
        episodes_count += 1
    
        if episodes_count % 10 == 0:
            mean_return = np.mean(rewards[-100:])
            std_return = np.std(rewards[-100:])
            recent_returns = rewards[-10:]
            returns_str = " ".join(map(str, recent_returns))
            print(
                f"Episode {episodes_count}, mean 100-episode return {mean_return:.2f} +-{std_return:.2f}, returns {returns_str}")
    
        if episodes_count > 100000:
            break
    agent1.save("q_table.npy")

def play():
    agent1 = QLearnAgent(epsilon = 0)
    agent2 = HumanAgent()
    env = Environment(agent2,True)
    agent1.load("q_table.npy")

    state = env.reset()
    done = False
    while not done:
        action = agent1.get_action(env, state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
    print("finished game")
    env.print_board()
    
    
def main():
    parser = argparse.ArgumentParser(description="Choose mode: train or play.")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--play", action="store_true", help="Run play mode")

    args = parser.parse_args()

    if args.train:
        train_q_learn()
    elif args.play:
        play()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
        
        