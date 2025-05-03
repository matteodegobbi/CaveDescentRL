import numpy as np
import random
import argparse
class RandomAgent:
    def move(self,env):
        return self.get_random_move(env)
        
    def get_random_move(self,env):
        available_coords = list(zip(*np.where(env.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            return None
        return random.choice(available)
        
class HumanAgent:
    def move(self,env):
        available_coords = list(zip(*np.where(env.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            print("No available move for human")
            return None
        return int(input())

        
class Environment:
    def __init__(self,agent2,prints_board=False):
        self.board = np.full((3, 3), -1)
        self.current_player = 0
        self.observation_space = 3 ** 9
        self.action_space = 9
        self.agent2 = agent2
        self.prints_board = prints_board
    
    def step(self, move: int):
        self.make_move(move, self.current_player)

        if self.is_done():
            # Current player won
            return self.get_state(), 1, True, False, {}

        if self.prints_board:
            self.print_board()
        self.current_player = self.get_opposite_player()
        other_move = self.agent2.move(self)
        if other_move is None:
            # Draw
            return self.get_state(), 0, True, False, {}
        self.make_move(other_move, self.current_player)

        if self.is_done():
            # The other player won
            return self.get_state(), -1, True, False, {}

        self.current_player = self.get_opposite_player()
        # next_state, reward, terminated, truncated, _
        return self.get_state(), 0, False, False, {}

    def make_move(self, move, player):
        one_hot_move = np.zeros(self.action_space)
        one_hot_move[move] = 1
        coords = (move // 3, move % 3)
        if self.board[coords] != -1:
            raise Exception("Already occupied")
        self.board[coords] = player


    def get_valid_moves(self):
        available_coords = list(zip(*np.where(self.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        moves = np.zeros(self.action_space)
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
    
    env = Environment(RandomAgent())
    state = env.reset()
    # np.random.seed()
    Q = np.zeros((env.observation_space, env.action_space))
    
    training = True
    epsilon = 0.1
    alpha = 0.5
    gamma = 0.5
    done = False
    
    rewards = []
    episodes_count = 0
    
    while training:
        # Perform episode
        state = env.reset()
    
        done = False
        reward_sum = 0
        while not done:
            if np.random.rand() < epsilon:
                action = env.sample_action()
            else:
                available_moves = env.get_valid_moves()
                q_values = Q[state]
                q_filtered = np.where(available_moves > 0, q_values, -np.inf)
                action = np.argmax(q_filtered)
    
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
            # Update the action-value estimates
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    
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
    np.save("q_table.npy", Q)
    
def play():
    Q = np.load("q_table.npy")
    env = Environment(HumanAgent(),True)
    state = env.reset()
    done = False
    while not done:
        available_moves = env.get_valid_moves()
        q_values = Q[state]
        q_filtered = np.where(available_moves > 0, q_values, -np.inf)
        print(q_filtered)
        action = np.argmax(q_filtered)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update the action-value estimates
        #Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

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
        
        