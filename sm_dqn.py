#!/bin/env python

import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import reorganize
import json 
import numpy as np
from argparse import ArgumentParser
import torch
from pathlib import Path


class CompactJSONEncoder(json.JSONEncoder):
    """
    Fast JSON encoder that:
      • Prints lists and numpy arrays of scalars compactly.
      • Prints small dicts on one line if they fit.
      • Pretty-prints larger or nested structures.
    """
    def __init__(self, *args, max_line_length=80, items_per_line=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_line_length = max_line_length
        self.items_per_line = items_per_line

    def encode(self, o):
        return self._encode_value(o, 0)

    # ---------------- internal helpers ----------------
    def _normalize(self, o):
        """Convert numpy objects to plain Python types."""
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return o

    def _encode_value(self, o, level):
        o = self._normalize(o)

        if isinstance(o, dict):
            return self._encode_dict(o, level)
        elif isinstance(o, list):
            return self._encode_list(o, level)
        else:
            return json.dumps(o)

    def _encode_dict(self, o, level):
        if not o:
            return "{}"

        # Try single-line compact representation
        inner = ", ".join(f"{json.dumps(k)}: {json.dumps(self._normalize(v))}"
                          for k, v in o.items()
                          if isinstance(v, (int, float, str, bool, type(None), np.generic)))
        if inner and len(inner) + 4 <= self.max_line_length:
            return f"{{{inner}}}"

        # Otherwise pretty-print recursively
        indent = " " * (self.indent * (level + 1)) if self.indent else ""
        pieces = []
        for k, v in o.items():
            key = json.dumps(k)
            val = self._encode_value(v, level + 1)
            pieces.append(f"{indent}{key}: {val}")
        joined = ",\n".join(pieces)
        outer_indent = " " * (self.indent * level) if self.indent else ""
        return f"{{\n{joined}\n{outer_indent}}}"

    def _encode_list(self, o, level):
        if not o:
            return "[]"

        o = [self._normalize(x) for x in o]

        # Flat scalar list?
        if all(isinstance(x, (int, float, str, bool, type(None))) for x in o):
            line = ", ".join(map(json.dumps, o))
            if len(line) <= self.max_line_length:
                return f"[{line}]"
            else:
                outer_indent = " " * (self.indent * level) if self.indent else ""
                inner_indent = " " * (self.indent * (level + 1)) if self.indent else ""
                lines = []
                for i in range(0, len(o), self.items_per_line):
                    chunk = ", ".join(map(json.dumps, o[i:i+self.items_per_line]))
                    lines.append(f"{inner_indent}{chunk}")
                joined = ",\n".join(lines)
                return f"[\n{joined}\n{outer_indent}]"
        else:
            # Non-scalar: pretty-print recursively
            inner_indent = " " * (self.indent * (level + 1)) if self.indent else ""
            outer_indent = " " * (self.indent * level) if self.indent else ""
            parts = [f"{inner_indent}{self._encode_value(x, level + 1)}" for x in o]
            joined = ",\n".join(parts)
            return f"[\n{joined}\n{outer_indent}]"

def make_env(args) -> "Env":
    env = rlcard.make('sergeant-major', {"sergeant-major.pad_state": True})
    return env

def set_agents(env: "Env", agent, args) -> None:
    random_agent = RandomAgent(num_actions=env.num_actions)
    agents = [agent, random_agent, random_agent]
    env.set_agents(agents)

def train(env, agent, args):
    def evaluate():
        payoff = 0
        for _ in range(eval_episodes):
            trajectories, payoffs = env.run(is_training=False)
            payoff += payoffs[0]
        payoff = payoff / eval_episodes
        print(f"\n\nepisode {episode}, payoff {payoff}\n")
        return payoff

    num_episodes = args.num_episodes or 10000
    eval_episodes = 100
    eval_every = 1000
    eval_payoffs = []
    for episode in range(num_episodes):
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for i in range(env.num_players):
            for trajectory in trajectories[i]:
                agent.feed(trajectory)
        if episode%eval_every == 0 or episode + 1 == num_episodes:
            eval_payoffs.append(evaluate())
    print(f"\n\npayoffs: {eval_payoffs}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--agent', type=str, default="DQNAgent")
    parser.add_argument('--num-episodes', type=int)
    parser.add_argument('--output', type=str)
    return parser.parse_args()

def make_agent(env, args):
    if args.agent == "DQNAgent":
        if args.input:
            torch.load(args.input)
            agent = DQNAgent.from_checkpoint(args.input, map_location="cpu")
        else:
            agent = DQNAgent(
                num_actions=env.num_actions, 
                state_shape=env.state_shape,
                mlp_layers=[64,64],
                replay_memory_size=20000,
                batch_size=32)
    else:
        raise NotImplementedError(f"Unknown agent {args.agent}")
    return agent


def save_agent(agent, args):
    if args.output:
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory {path}")
        agent.save_checkpoint(path)


args = parse_args()
print(args)
env = make_env(args)
agent = make_agent(env, args)
set_agents(env, agent, args)
train(env, agent, args)
save_agent(agent, args)
