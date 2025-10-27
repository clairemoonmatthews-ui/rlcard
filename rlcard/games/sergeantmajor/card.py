from rlcard.games.base import Card
from typing import List

class SergeantMajorCard(Card):
    @classmethod
    def get_deck(cls) -> List["SergeantMajorCard"]:
        return [
            SergeantMajorCard(suit=suit, rank=rank) 
            for suit in SergeantMajorCard.suits[:4] 
            for rank in SergeantMajorCard.ranks
        ]
