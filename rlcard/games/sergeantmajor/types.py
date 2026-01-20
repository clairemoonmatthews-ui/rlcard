from typing import List, Tuple
from rlcard.games.base import Card
from dataclasses import dataclass

PlayerId = int
Trick = List[Tuple[PlayerId, Card]]
Tricks = List[Trick]
Hand = List[Card]
Actions = List[Card]
Suit = int


