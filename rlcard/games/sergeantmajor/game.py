from collections import OrderedDict
from typing import Any, Dict, Tuple
import numpy as np
from rlcard.games.sergeantmajor.judger import SergeantMajorJudger
from rlcard.games.sergeantmajor.round import SergeantMajorRound
from rlcard.games.sergeantmajor.types import PlayerId
from rlcard.games.sergeantmajor.state import PlayerState


class SergeantMajorGame:
    """
    Main game class for Sergeant Major.
    Coordinates rounds and provides the interface for external interaction.
    """
    
    def __init__(self, allow_step_back: bool = False) -> None:
        """
        Initialize a Sergeant Major game.
        
        Args:
            allow_step_back: Whether to allow stepping back in the game history
        """
        self.allow_step_back = allow_step_back
        self.np_random = np.random.default_rng()
        self.num_players = 3
    
    def init_game(self) -> Tuple[PlayerState, PlayerId]:
        """
        Initialize a new game by creating a round and dealing cards.
        
        Returns:
            Initial state observation for the first player
        """
        self.round = SergeantMajorRound(self.np_random, self.num_players)
        return (self.get_state(self.get_player_id()), self.get_player_id())
    
    def step(self, action: Any) -> Tuple[PlayerState, int]:
        """
        Execute one action in the game.
        
        Args:
            action: The action to execute (typically a Card object)
            
        Returns:
            Tuple of (state observation for next player, next player id)
        """
        self.round.proceed_round(action)
        player_id = self.round.current_player_id
        player_state = self.round.get_state(player_id)
        return (player_state, player_id)
    
    def get_state(self, player_id: int) -> PlayerState:
        """
        Get the state observation for a specific player.
        
        Args:
            player_id: The player whose perspective to get
            
        Returns:
            Dictionary containing the player's observation
        """
        return self.round.get_state(player_id)
    
    def get_rlcard_state(self) -> Dict:
        """
        Get the state observation for the current player.
            
        Returns:
            Dictionary containing the player's observation
        """
        player_state = self.get_state(self.get_player_id())
        obs, raw_obs = player_state.to_tokens(False)
        legal_actions = {}
        raw_legal_actions = []
        legal_actions_cards = self.round.get_legal_actions(self.get_player_id())
        legal_actions_int = [c.get_index() for c in legal_actions_cards]
        for action in legal_actions_int:
            legal_actions[(action)] = None
            raw_legal_actions.append(str(action))
        legal_actions = OrderedDict(legal_actions)

        extracted_self = {
            'obs': obs, 
            'raw_obs': raw_obs,
            'legal_actions': legal_actions, 'raw_legal_actions': raw_legal_actions}
        return extracted_self
    
    def get_payoffs(self) -> np.array:
        return np.array(SergeantMajorJudger.judge_game(self))/16

    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        
        Returns:
            Number of players (always 3 for Sergeant Major)
        """
        return self.num_players
    
    def get_num_actions(self) -> int:
        """
        Get the total number of possible actions.
        
        Returns:
            Total number of possible actions (52 for a standard deck)
        """
        return 52
    
    def get_player_id(self) -> int:
        """
        Get the current player's id.
        
        Returns:
            ID of the player whose turn it is
        """
        return self.round.current_player_id
    
    def is_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game has ended, False otherwise
        """
        return len(self.round.tricks) == 16
    
    @classmethod
    def from_player_state (cls, player_state:PlayerState, allow_step_back: bool = False):
        game = cls(allow_step_back)
        game.round = SergeantMajorRound.from_player_state(player_state, game.np_random)
        return game
    
    @classmethod
    def from_rlcard_state (cls, state: Dict, allow_step_back: bool = False):
        obs = state["obs"]
        player_state = PlayerState.from_tokens(obs)
        return cls.from_player_state(player_state, allow_step_back)

    
    
