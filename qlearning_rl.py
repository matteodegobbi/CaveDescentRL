import os

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
    def get_action(self, env, state):
        available_coords = list(zip(*np.where(env.board == -1)))
        available = [r * 3 + c for r, c in available_coords]
        if not available:
            print("No available move for human")
            return None
        while True:
            try:
                move = int(input(f"Enter your move (0-8) for player {env.symbol(env.current_player)}: "))
                if move in available:
                    return move
                else:
                    print("Invalid move. Choose an empty cell.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 8.")
            except Exception as e:
                print(f"An error occurred: {e}")

class QLearnAgent:

    def __init__(self,epsilon = 0.1,alpha = 0.5,gamma = 0.5):
        self.Q = np.zeros((observation_space, action_space))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self,env,state):
        available_moves_indices = env.get_valid_move_indices()
        if not available_moves_indices:
            return None

        if np.random.rand() < self.epsilon:
            action = random.choice(available_moves_indices)
        else:
            q_values = self.Q[state]
            q_filtered = {move: q_values[move] for move in available_moves_indices}
            max_q = -np.inf
            best_actions = []
            for move, q in q_filtered.items():
                if q > max_q:
                    max_q = q
                    best_actions = [move]
                elif q == max_q:
                    best_actions.append(move)
            action = random.choice(best_actions)
        return action

    def learn(self, state, action, next_state, reward, done):
        if done:
            target = reward
        else:
            next_q_values = self.Q[next_state]
            max_next_q = np.max(next_q_values)
            target = reward + self.gamma * max_next_q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


    def load(self, path):
        if os.path.exists(path):
            self.Q = np.load(path)
            print(f"Q-table loaded from {path}")
        else:
            print(f"Warning: Q-table file not found at {path}. Starting with empty table.")

    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table saved to {path}")


class Environment:
    def __init__(self, prints_board=False):
        self.board = np.full((3, 3), -1)
        self.current_player = 0
        self.prints_board = prints_board

    def step(self, move: int):
        if move is None or not self.is_valid_move(move):
            # This indicates an error in the agent logic or training loop
            raise ValueError(f"Invalid move {move} attempted by player {self.current_player} on board:\n{self.board}")

        # Make move
        coords = (move // 3, move % 3)
        self.board[coords] = self.current_player

        if self.prints_board: self.print_board()

        # Check win
        won = self.check_win(self.current_player)
        if won:
            return self.get_state(), 1, True

        if self.is_draw():
            return self.get_state(), 0, True

        self.current_player = self.get_opposite_player()
        return self.get_state(), 0, False

    def is_valid_move(self, move):
        if move < 0 or move >= action_space:
            return False
        coords = (move // 3, move % 3)
        return self.board[coords] == -1

    def is_draw(self):
        # Draw if the board is full and no one has won (win check happens first)
        return not np.any(self.board == -1)


    def get_valid_move_indices(self):
         available_coords = list(zip(*np.where(self.board == -1)))
         return [r * 3 + c for r, c in available_coords]

    def get_valid_moves_mask(self): # Returns a boolean mask
        return self.board.flatten() == -1

    def reset(self):
        self.board = np.full((3, 3), -1)
        self.current_player = 0
        return self.get_state()

    def get_opposite_player(self):
        return 1 - self.current_player

    def check_win(self, player):
        # Check rows, columns, and diagonals for the specified player
        for i in range(3):
            if np.all(self.board[i, :] == player): return True
            if np.all(self.board[:, i] == player): return True
        if np.all(np.diag(self.board) == player): return True
        if np.all(np.diag(np.fliplr(self.board)) == player): return True
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

# --- Training Function (Self-Play) ---
def train_self_play(episodes=500000, # Increased episodes for better convergence
                    epsilon_start=1.0,  # Start with full exploration
                    epsilon_end=0.01,   # Keep a small amount of exploration
                    epsilon_decay=0.99999, # Slower decay rate
                    alpha=0.01,        # Learning rate
                    gamma=0.99,       # Discount factor, prioritize future rewards slightly more
                    print_every=5000, # Print progress less frequently for long runs
                    save_every=20000,
                    preload_models=False): # Save checkpoints periodically

    # Create two Q-learning agents
    agent_0 = QLearnAgent(epsilon=epsilon_start, alpha=alpha, gamma=gamma)
    agent_1 = QLearnAgent(epsilon=epsilon_start, alpha=alpha, gamma=gamma)
    agents = [agent_0, agent_1]

    # Try loading existing Q-tables
    if preload_models:
        agent_0.load("q_table_agent0.npy")
        agent_1.load("q_table_agent1.npy")

    env = Environment(prints_board=False) # Don't print board during mass training

    # --- Store outcome of each episode for accurate reporting ---
    episode_outcomes = [] # 0: Agent 0 win, 1: Agent 1 win, 2: Draw

    current_epsilon = epsilon_start

    print(f"Starting training for {episodes} episodes...")
    print(f"Parameters: eps_start={epsilon_start}, eps_end={epsilon_end}, eps_decay={epsilon_decay}, alpha={alpha}, gamma={gamma}")

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        last_transition = {0: None, 1: None} # Store (s, a) for each agent's last move
        episode_winner = -1 # Track winner for this specific episode (-1: undecided, 0: agent0, 1: agent1, 2: draw)

        while not done:
            # Determine current player and agent
            current_player_idx = env.current_player
            current_agent = agents[current_player_idx]
            current_agent.epsilon = current_epsilon

            action = current_agent.get_action(env, state)

            # TODO: maybe delete this
            if action is None or not env.is_valid_move(action):
                print(f"Warning: Episode {episode}, Player {current_player_idx}. Invalid action ({action}) attempted or no moves left unexpectedly. Forcing draw.")
                valid_moves = env.get_valid_move_indices()
                print(f"Valid moves were: {valid_moves}")
                print(f"Current Board:\n{env.board}")
                episode_winner = 2
                done = True
                break

            # --- Learning Step for the Opponent (from their previous move) ---
            # If the opponent made a move previously in this episode, they learn now.
            # Their previous action led to the current 'state'. Reward is 0 as game wasn't over then.

            opponent_idx = 1 - current_player_idx
            if last_transition[opponent_idx] is not None:
                prev_state_opp, prev_action_opp = last_transition[opponent_idx]
                agents[opponent_idx].learn(prev_state_opp, prev_action_opp, state, reward=0, done=False)

            # Store current state and action *before* stepping the environment
            last_transition[current_player_idx] = (state, action)
            last_transition[opponent_idx] = None # Clear opponent's transition until they move again

            next_state, reward, done = env.step(action) # reward for current

            # --- Learning Step upon Game End ---
            if done:
                # The player who just moved (current_agent) learns from the outcome
                current_agent.learn(state, action, next_state, reward, done=True)

                # The opponent also learns from their previous action (if it exists),
                # It obtains the negative reward wrt the current_agent
                if last_transition[opponent_idx] is not None:
                     prev_state_opp, prev_action_opp = last_transition[opponent_idx]
                     opponent_reward = -reward
                     agents[opponent_idx].learn(prev_state_opp, prev_action_opp, next_state, opponent_reward, done=True)

                if reward == 1: # current_player_idx won
                    episode_winner = current_player_idx
                elif reward == 0: # Draw
                    episode_winner = 2 # 2 for draw

            state = next_state

        assert episode_winner != -1
        episode_outcomes.append(episode_winner)

        # Decay epsilon after each episode
        current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay)
        agents[0].epsilon = current_epsilon
        agents[1].epsilon = current_epsilon

        if episode % print_every == 0:
            print(f"--- Episode {episode}/{episodes} | Epsilon: {current_epsilon:.5f} ---")
            if len(episode_outcomes) >= print_every:
                window_outcomes = episode_outcomes[-print_every:]
                wins_0 = window_outcomes.count(0)
                wins_1 = window_outcomes.count(1)
                draws = window_outcomes.count(2)
                actual_total_in_window = len(window_outcomes)
                assert actual_total_in_window == print_every, "Error: actual_total_in_window != print_every"

                if actual_total_in_window > 0:
                   print(f"  Stats over last {actual_total_in_window} episodes:")
                   print(f"    Agent 0 Wins (X): {wins_0:5d} ({wins_0/actual_total_in_window:6.1%})")
                   print(f"    Agent 1 Wins (O): {wins_1:5d} ({wins_1/actual_total_in_window:6.1%})")
                   print(f"    Draws:            {draws:5d} ({draws/actual_total_in_window:6.1%})")
                else:
                    print("  Error: No data in the reporting window.")
            else:
                 print(f"  (Collecting more data... Need {print_every} episodes for first stats report)")

        if episode % save_every == 0:
            print(f"--- Saving Q-tables at episode {episode} ---")
            agents[0].save("q_table_agent0.npy")
            agents[1].save("q_table_agent1.npy")

    # Final save after all episodes are done
    print("--- Training finished. Saving final Q-tables. ---")
    agents[0].save("q_table_agent0.npy")
    agents[1].save("q_table_agent1.npy")
    print("Training complete.")


def play(q_agent_file="q_table_agent0.npy", agent_plays_as=1):
    agent_q = QLearnAgent(epsilon=0)
    agent_q.load(q_agent_file)
    agent_human = HumanAgent()
    env = Environment(prints_board=True)

    if agent_plays_as == 0:
        agents = [agent_q, agent_human]
        print("Q-Agent plays as X (starts)")
    else:
        agents = [agent_human, agent_q]
        print("Q-Agent plays as O (goes second)")


    state = env.reset()
    done = False
    env.print_board()

    while not done:
        current_player_idx = env.current_player
        current_agent = agents[current_player_idx]

        action = current_agent.get_action(env, state)

        if action is None:
            print("No available moves left.")
            break

        next_state, reward, done = env.step(action)


        if done:
            if reward == 1:
                print(f"Player {env.symbol(current_player_idx)} ({type(current_agent).__name__}) wins!")
            elif reward == 0:
                print("It's a draw!")
            break
        state = next_state
    print("\n--- Finished ---")

def main():
    parser = argparse.ArgumentParser(
        description="Train Q-learning agents via self-play or play against a trained agent.")
    parser.add_argument("--train", action="store_true", help="Run self-play training mode")
    parser.add_argument("--play", action="store_true", help="Run play mode (Q-agent vs Human)")
    parser.add_argument("--episodes", type=int, default=500000, help="Number of episodes for training")
    parser.add_argument("--agent-file", type=str, default="q_table_agent0.npy",
                        help="Q-table file to load for playing (or save during training)")
    parser.add_argument("--agent-player", type=int, default=0, choices=[0, 1],
                        help="Which player the Q-agent should be in play mode (0=X, 1=O)")
    parser.add_argument("--print-every", type=int, default=1000, help="Frequency of printing training progress")
    parser.add_argument("--save-every", type=int, default=10000,
                        help="Frequency of saving Q-tables during training")
    parser.add_argument("--preload", action="store_true", help="Run self-play training mode")

    args = parser.parse_args()

    if args.train:
        train_self_play(
            episodes=args.episodes,
            print_every=args.print_every,
            save_every=args.save_every,
            preload_models=args.preload
        )
    elif args.play:
        play(q_agent_file=args.agent_file, agent_plays_as=args.agent_player)
    else:
        print("Please specify --train or --play")
        parser.print_help()

if __name__ == "__main__":
    main()