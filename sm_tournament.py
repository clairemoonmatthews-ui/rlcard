#!/bin/env python
import numpy as np
import rlcard
from rlcard.agents import random_agent
from rlcard.agents.mcts_agent import MCTSAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sergeantmajor_agent import HeuristicAgent
from rlcard.agents.transformer_agent import TransformerAgent
from rlcard.envs.env import Env
from rlcard.games.sergeantmajor.game import SergeantMajorGame
from rlcard.utils.utils import tournament

import logging
from tqdm import tqdm
from multiprocessing import Pool
import os

logging.basicConfig(level=logging.WARNING)

# logging.basicConfig(level=logging.INFO)
competitor = "mcts"
opponent = "heuristic"  # random, heuristic, model
model = "models/model_20260127T020706Z.pt"
short_model = model.split("/")[-1].split(".")[0]
game_class = SergeantMajorGame
game_class_name = 'sergeant-major'
mcts_num_simulations = 100
n_games = 100


transformer_agent = None


def get_transformer_agent():
    """Singleton pattern to load the transformer agent only once per process."""
    global transformer_agent
    if transformer_agent is None:
        # Stagger loads slightly to reduce I/O contention
        import time
        import random
        sleep_time = random.random() * 0.5
        # print(
        #     f"Loading transformer agent from {model} (sleeping for {sleep_time:.1f}s to reduce contention)...")
        time.sleep(sleep_time)
        start_time = time.time()
        transformer_agent = TransformerAgent.load(model, train=False)
        # print(f"Transformer agent loaded in {time.time() - start_time:.1f}s")
    return transformer_agent


def make_agent(key: str):
    env = rlcard.make(game_class_name)  # just for num_actions
    match key:
        case "random":
            return RandomAgent(num_actions=env.num_actions), f"random"
        case "heuristic":
            return HeuristicAgent(), f"heuristic"
        case "transformer":
            return get_transformer_agent(), f"transformer({short_model})"
        case "mcts":
            transformer_agent = get_transformer_agent()
            name = f"mcts({short_model}, {mcts_num_simulations})"
            return MCTSAgent(agent=transformer_agent, game_class=game_class, num_simulations=mcts_num_simulations), name
        case _:
            assert False, f"Unknown competitor {key}"


def run_single_game(seed):
    """Run a single game and return 1 if competitor won, 0 otherwise."""
    import time
    import torch

    # Limit PyTorch to 1 thread per worker to avoid contention
    torch.set_num_threads(1)

    start = time.time()
    # print(f"[{seed}] Starting game")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # print(f"[{seed}] Creating agents (elapsed: {time.time()-start:.1f}s)")
    # Create fresh agents for this game
    competitor_agent, _ = make_agent(competitor)
    opponent_agent, _ = make_agent(opponent)

    # print(f"[{seed}] Agents created (elapsed: {time.time()-start:.1f}s)")
    # Random position
    position = np.random.randint(3)

    # Create environment and run game
    env = rlcard.make('sergeant-major')
    agents = [competitor_agent if i == position else opponent_agent
              for i in range(env.num_players)]
    env.set_agents(agents)

    # print(f"[{seed}] Running game with competitor at position {position} (elapsed: {time.time()-start:.1f}s)")
    trajectories, payoffs = env.run(is_training=False)

    # print(f"[{seed}] Game finished (elapsed: {time.time()-start:.1f}s)")
    # Check if competitor won
    max_payoff = max(payoffs)
    win = int(payoffs[position] == max_payoff)/sum(p == max_payoff for p in payoffs)
    # print(f"[{seed}] Result: {'Win' if win else 'Loss'} - payoffs={payoffs}")
    return dict(win=win, payoffs=payoffs, position=position, seed=seed, duration=time.time()-start)


if __name__ == '__main__':
    # Get number of CPUs to use
    n_processes = os.cpu_count()

    print(f"Running {n_games} games using {n_processes} processes...")

    # Create agent names for display (DON'T load the model here - let each worker load its own copy)
    competitor_name = {
        "random": "random",
        "heuristic": "heuristic",
        "transformer": f"transformer({short_model})",
        "mcts": f"mcts({short_model}, {mcts_num_simulations})"
    }[competitor]

    opponent_name = {
        "random": "random",
        "heuristic": "heuristic",
        "transformer": f"transformer({short_model})",
        "mcts": f"mcts({short_model}, {mcts_num_simulations})"
    }[opponent]

    print(f"Competitor: {competitor_name}, Opponent: {opponent_name}")

    # Run games in parallel with progress bar that shows win rate
    wins = 0
    total_payoff = 0
    total_duration = 0
    max_duration = 0
    min_duration = float('inf')
    with Pool(processes=n_processes) as pool:
        # Use imap_unordered to get results as they complete (not in order)
        with tqdm(total=n_games) as pbar:
            for result in pool.imap_unordered(run_single_game, range(n_games), chunksize=1):
                wins += result['win']
                total_payoff += result['payoffs'][result['position']]
                total_duration += result['duration']
                max_duration = max(max_duration, result['duration'])
                min_duration = min(min_duration, result['duration'])
                pbar.update(1)
                pbar.set_postfix({"wins": f"{wins / pbar.n:.2%}", "payoff": f"{total_payoff / pbar.n:.2f}",
                                 "t": f"{total_duration / pbar.n:.2f}s"})

    # Calculate final win rate
    win_rate = wins / n_games
    print(
        f"Competitor={competitor_name}, Opponent={opponent_name}, Win rate: {win_rate:.2%}, Avg payoff: {total_payoff / n_games:.2f}, Avg duration: {total_duration / n_games:.2f}s, Max duration: {max_duration:.2f}s, Min duration: {min_duration:.2f}s")
