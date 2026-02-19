from math import sqrt
import time
from typing import Dict, Optional, Tuple, Type
import logging

import numpy as np
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey

from rlcard.agents.transformer_agent import TransformerAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

RLCardState = Dict  # contains "obs" and "legal_actions"


class Node:
    """Simple class to represent nodes in an MCTS tree."""

    def __init__(self, num_actions: int, num_players: int, parent: Optional['Node'] = None, action: Optional[int] = None):
        self.parent = parent
        self.children = {}
        self.visits = np.zeros(num_actions, dtype=np.int32)
        self.values = np.zeros([num_actions, num_players], dtype=np.float32)
        self.action = action

    def best_action(self) -> int:
        """Selects the best action using visit counts."""
        return np.argmax(self.visits)


class MCTSAgent:
    """Agent using MCTS and another agent"""

    def __init__(self, agent: TransformerAgent, game_class: Type, num_simulations: int = 100, c_puct=sqrt(2), cache_size=10000):
        self.agent = agent
        self.game_class = game_class
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.use_raw = False
        self._policy_cache = LRUCache(maxsize=cache_size)

    def step(self, state: RLCardState) -> int:
        """
        Called during gameplay to select an action.

        Args:
            state: Dict with 'obs' (token sequence) and 'legal_actions' (list of valid card indices)

        Returns:
            action: Integer in range [0, 51] representing card to play
        """
        logger.info(f"{self} MCTS: Taking a step")
        legal_actions = list(state['legal_actions'].keys())
        assert len(legal_actions) > 0, "No legal actions available"
        if len(legal_actions) == 1:
            return legal_actions[0]
        root = Node(num_actions=self.agent.actions,
                    num_players=self.agent.nplayers)
        for i in range(self.num_simulations):
            # start_time = time.time()
            self.simulate(root, state)
            # logger.info(
            #     f"{self} MCTS: Simulation {i+1}/{self.num_simulations} took {time.time() - start_time:.2f}s")
        action = root.best_action()
        assert action in legal_actions, f"Selected action {action} not in legal actions {legal_actions}"
        logger.info(
            f"{self} MCTS: Selected action {action} after {self.num_simulations} simulations (visits: {root.visits[action]})")
        return action

    def eval_step(self, state: Dict) -> Tuple[int, Dict]:
        """
        Called during evaluation - returns action and info dict.

        Args:
            state: Same as step()

        Returns:
            action: Selected action
            info: Dict with any debugging/analysis info (can be empty)
        """
        action = self.step(state)
        return action, {}

    @cachedmethod(cache=lambda self: self._policy_cache, key=lambda self, state: hashkey(tuple(state['obs'])))
    def _get_policy(self, state: RLCardState) -> np.ndarray:
        """Get policy with LRU caching based on obs. State is passed through unchanged."""
        return self.agent.get_policy(state)

    def pick_explore_action(self, node: Node, game):
        # PUCT = values_for_current_player + c_puct * priors * sqrt(total_visits + 1) / (1 + visits)
        state = game.get_rlcard_state()
        current_player = game.get_player_id()
        legal_actions = list(state['legal_actions'].keys())
        assert len(legal_actions) > 0, f"No legal actions available at node"
        values = node.values[legal_actions, current_player]
        visits = node.visits[legal_actions]
        total_visits = np.sum(visits)
        policy = self._get_policy(state)
        # Check that policy sums to 1 and is between 0 and 1
        assert np.all(policy >= 0) and np.all(
            policy <= 1), f"Policy values out of range: {policy}"
        assert np.isclose(
            np.sum(policy), 1), f"Policy does not sum to 1: {policy}"

        # Probability for non legal actions should be 0
        illegal_actions = list(set(range(len(policy))) - set(legal_actions))
        assert np.all(policy[illegal_actions] ==
                      0), f"Illegal actions have non-zero policy"
        priors = policy[legal_actions]
        puct = values + self.c_puct * priors * \
            np.sqrt(total_visits + 1) / (1 + visits)
        i = np.argmax(puct)
        action = legal_actions[i]
        assert action in legal_actions, f"Selected action {action} not in legal actions {legal_actions}"
        return action

    def simulate(self, root: Node, state: RLCardState):
        """
        Run one step of simulation of MCTS.

        Arguments:
            root: Node we're starting with
            obs: Observations from game        
        """
        # 1. Determinize: Create a realized game state from the observations.
        game = self.game_class.from_rlcard_state(state)

        # 2. Selection: Starting at the root, use PUCT to select actions until
        #   either you reach an unexpanded node, or you reach a terminal state.
        node = root
        while True:
            if game.is_over():
                break
            action = self.pick_explore_action(node, game)

            if action not in node.children:
                # 3. Expansion: Unless you are at a terminal state, create one new node.
                new_node = Node(parent=node, action=action,
                                num_actions=self.agent.actions, num_players=self.agent.nplayers)
                node.children[action] = new_node
                node = new_node
                game.step(action)
                break
            node = node.children[action]
            game.step(action)

        # 4. Evaluate: At terminal states, find the actual payoffs;
        #    otherwise use the transformer agent to estimate value.
        if game.is_over():
            payoffs = game.get_payoffs()
        else:
            payoffs = self.agent.get_value(game.get_rlcard_state())

        # Check that values are between 0 and 1 and sum to 1
        assert np.all(payoffs >= 0) and np.all(
            payoffs <= 1), f"Payoff values out of range: {payoffs}"
        assert np.isclose(
            np.sum(payoffs), 1), f"Payoffs do not sum to 1: {payoffs}"

        # 5. Backpropagation: Adjust all the nodes back to the root with visit count
        #    and updated Q-value.
        while node != root:
            node.parent.values[node.action, :] = (
                node.parent.visits[node.action] * node.parent.values[node.action, :] + payoffs) / (node.parent.visits[node.action] + 1)
            # Check that values are between 0 and 1 and sum to 1
            assert np.all(node.parent.values[node.action, :] >= 0) and np.all(
                node.parent.values[node.action, :] <= 1), f"Updated values out of range: {node.parent.values[node.action, :]}"
            assert np.isclose(np.sum(
                node.parent.values[node.action, :]), 1), f"Updated values do not sum to 1: {node.parent.values[node.action, :]}"
            node.parent.visits[node.action] += 1
            node = node.parent
