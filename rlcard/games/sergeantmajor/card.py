from rlcard.games.base import Card
from typing import List

class SergeantMajorCard(Card):
    _deck = [
        SergeantMajorCard(suit=suit, rank=rank) 
        for suit in Card.suits[:4] 
        for rank in Card.ranks
    ]
    @classmethod
    def get_deck(cls) -> List["SergeantMajorCard"]:
        return cls._deck.copy()

    @classmethod
    def from_index(cls, index: int) -> Card:
        """Creates a card from an action id
        
        Args:
            index: Integer between 0 and 51
            
        Returns: 
            card
        """
        return SergeantMajorCard._deck[index]

    def get_index(self) -> int:
        """Returns the index associated with a card
        
        Returns:
            index: Integer between 0 and 51
        """
        for i, card in enumerate(SergeantMajorCard._deck):
            if self == card:
                return i
        raise IndexError()