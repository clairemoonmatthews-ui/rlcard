from collections import OrderedDict
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional
import numpy as np
from rlcard.envs import Env
from rlcard.games.base import Card
from rlcard.games.sergeantmajor import SergeantMajorGame
from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.judger import SergeantMajorJudger
from rlcard.games.sergeantmajor.types import PlayerId, PlayerState, Trick

class SergeantMajorEnv(Env):
    """
    Sergeant Major Environment wrapper for RLCard.
    Provides the standardized environment interface for agents to interact with the game.
    """


    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize a Sergeant Major environment.
        
        Args:
            config: Configuration dictionary containing environment settings
        """
        self.game = SergeantMajorGame()
        self.name = "sergeant-major"
        super().__init__(config)
        self.state_shape = [[self.max_state_length]] # maximum length
        self.action_shape = [None]
        self.pad_state = config.get("sergeant-major.pad_state", False)
    
    def _extract_state(self, state:PlayerState):
        obs, raw_obs = state.to_tokens(self.pad_state)
        legal_actions = {}
        raw_legal_actions = []
        for action in self.legal_actions:
            legal_actions[(action.get_index())] = None
            raw_legal_actions.append(str(action))
        legal_actions = OrderedDict(legal_actions)

        extracted_self = {
            'obs': obs, 
            'raw_obs': raw_obs,
            'legal_actions': legal_actions, 'raw_legal_actions': raw_legal_actions}
        return extracted_self
    
    def _decode_action(self, action_id: int) -> Any:
        """
        Decode an action ID into the game's action format.
        Converts integer action indices to Card objects or other game actions.
        
        Args:
            action_id: Integer action ID from the agent
            
        Returns:
            Game-specific action (typically a Card object)
        """
        return SergeantMajorCard.from_index(action_id)
    
    def _get_legal_actions(self) -> List[int]:
        """
        Get the list of legal action IDs for the current player.
        
        Returns:
            List of integer action IDs that are currently legal
        """
        legal_actions_cards = self.game.round.get_legal_actions(self.game.get_player_id())
        legal_actions_int = [c.get_index() for c in legal_actions_cards]
        return legal_actions_int
    
    def get_payoffs(self) -> List[float]:
        """
        Get the final payoffs for all players.
        
        Returns:
            List of payoffs (one per player)
        """
        return np.array(SergeantMajorJudger.judge_game(self.game))
    
    def _load_model(self) -> None:
        """
        Load any pretrained models if specified in config.
        Optional method for loading saved models.
        """
        pass