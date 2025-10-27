from typing import List
from rlcard.games.sergeantmajor.game import SergeantMajorGame


class SergeantMajorJudger:
    """
    Judges the game and computes final payoffs for each player.
    """
    
    @staticmethod
    def judge_game(game: SergeantMajorGame) -> List[int]:
        """
        Compute the final payoffs for all players.
        
        Args:
            game: The completed game to judge
            
        Returns:
            List of payoffs (one per player), typically the number of tricks won
        """
        pass