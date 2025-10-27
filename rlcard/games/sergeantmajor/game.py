from typing import Tuple
import numpy as np
from rlcard.games.sergeantmajor.round import SergeantMajorRound
from rlcard.games.sergeantmajor.types import PlayerState


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
        pass
    
    def init_game(self) -> PlayerState:
        """
        Initialize a new game by creating a round and dealing cards.
        
        Returns:
            Initial state observation for the first player
        """
        pass
    
    def step(self, action: Any) -> Tuple[PlayerState, int]:
        """
        Execute one action in the game.
        
        Args:
            action: The action to execute (typically a Card object)
            
        Returns:
            Tuple of (state observation for next player, next player id)
        """
        pass
    
    def get_state(self, player_id: int) -> PlayerState:
        """
        Get the state observation for a specific player.
        
        Args:
            player_id: The player whose perspective to get
            
        Returns:
            Dictionary containing the player's observation
        """
        pass
    
    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        
        Returns:
            Number of players (always 3 for Sergeant Major)
        """
        pass
    
    def get_num_actions(self) -> int:
        """
        Get the total number of possible actions.
        
        Returns:
            Total number of possible actions (52 for a standard deck)
        """
        pass
    
    def get_player_id(self) -> int:
        """
        Get the current player's id.
        
        Returns:
            ID of the player whose turn it is
        """
        pass
    
    def is_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game has ended, False otherwise
        """
        pass