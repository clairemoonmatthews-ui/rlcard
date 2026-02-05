from cmath import sqrt
from typing import Dict, Optional, Type

import numpy as np

from rlcard.agents.transformer_agent import TransformerAgent


class Node:
    def __init__(self, num_actions:int, num_players:int, parent:Optional['Node']=None):
        self.parent = parent
        self.children = {}
        self.visits = np.zeros(num_actions, dtype=np.int32)
        self.values = np.zeros([num_actions, num_players], dtype=np.float32)

    def best_action(self) -> int:
        return np.argmax(self.visits)

class MCTSAgent:
    def __init__(self, agent:TransformerAgent, game_class:Type, num_simulations:int=100, c_puct=sqrt(2)):

        self.agent = agent
        self.game_class = game_class
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    # def pick_move(self, game, ):

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
        pass