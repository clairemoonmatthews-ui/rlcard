from cmath import sqrt
from typing import Dict, Optional, Type

import numpy as np

from rlcard.agents.transformer_agent import TransformerAgent


class Node:
    """Simple class to represent nodes in an MCTS tree."""
    def __init__(self, num_actions:int, num_players:int, parent:Optional['Node']=None):
        self.parent = parent
        self.children = {}
        self.visits = np.zeros(num_actions, dtype=np.int32)
        self.values = np.zeros([num_actions, num_players], dtype=np.float32)

    def best_action(self) -> int:
        """Selects the best action using visit counts."""
        return np.argmax(self.visits)

class MCTSAgent:
    """Agent using MCTS and another agent"""
    def __init__(self, agent:TransformerAgent, game_class:Type, num_simulations:int=100, c_puct=sqrt(2)):
        self.agent = agent
        self.game_class = game_class
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def step(self, state:Dict) -> int:
        """
        Called during gameplay to select an action.
        
        Args:
            state: Dict with 'obs' (token sequence) and 'legal_actions' (list of valid card indices)
        
        Returns:
            action: Integer in range [0, 51] representing card to play
        """
        legal_actions = state['legal_actions']
        obs = state['obs']  # The token sequence
        root = Node(num_actions=self.agent.actions, num_players=self.agent.nplayers)
        for i in range(self.num_simulations):
            self.simulate(root, obs)
        return root.best_action()

    def simulate(root:Node, obs:np.array):
        """
        Run one step of simulation of MCTS.

        Arguments:
            root: Node we're starting with
            obs: Observations from game        
        """
        pass

        # We are implementing Alpha Zero style MCTS with imperfect information.
        # We need to do the following five steps:
        # 1. Determinize: Create a realized game state from the observations.
        # 2. Selection: Starting at the root, use PUCT to select actions until
        #    either you reach an unexpanded node, or you reach a terminal state.
        #    We get the policy prior from the transformer agent.
        # 3. Expansion: Unless you are at a terminal state, create one new node.
        # 4. Evaluate: At terminal states, find the actual payoffs; 
        #    otherwise use the transformer agent to estimate value.
        # 5. Backpropagation: Adjust all the nodes back to the root with visit count 
        #    and updated Q-value.

        # Hmm.  Might want to change this method to take an rlcard state dict 
        # instead of obs.

        # Some useful methods:
        # self.agent.get_policy() - not yet implemented
        # self.agent.get_value()
        # self.game_class.from_rlcard_state() 
        # game.is_over()
        # game.step()
        # game.get_rlcard_state() - returns both obs and legal_actions
        # game.get_payoffs() - not yet implemented, but see SergeantMajorEnv; think about scaling 
        # game.get_player_id()

        # PUCT = values_for_current_player + c_puct * priors * sqrt(total_visits + 1) / (1 + visits)
        # Note that values, priorts and visits should be indexed by legal actions.

