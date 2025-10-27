from typing import List, Tuple
from rlcard.games.base import Card
from dataclasses import dataclass

PlayerId = int
Trick = List[Tuple[PlayerId, Card]]
Tricks = List[Trick]
Hand = List[Card]
Actions = List[Card]

@dataclass
class PlayerState:
    """Type definition for a player's observable state."""
    current_player: PlayerId
    hand: Hand = []
    current_trick: Trick = []
    tricks: Tricks = []
    tricks_won: int = 0
    legal_actions: Actions = []

