from rlcard.games.base import Card
from typing import List

class SergeantMajorCard(Card):
    
    @classmethod
    def get_deck(cls) -> List["SergeantMajorCard"]:
        return _deck.copy()

    @classmethod
    def from_index(cls, index: int) -> Card:
        """Creates a card from an action id
        
        Args:
            index: Integer between 0 and 51
            
        Returns: 
            card
        """
        return _deck[index]

    def get_index(self) -> int:
        """Returns the index associated with a card
        
        Returns:
            index: Integer between 0 and 51
        """
        for i, card in enumerate(_deck):
            if self == card:
                return i
        raise IndexError()
    
    @property
    def rank_index(self) -> int:
        """Returns the index associated with a cards rank, ace is high so will be 13
        
        Returns:
            index: Integer between 1 and 13
        """
        i = Card.valid_rank.index(self.rank)
        if i == 0:
            i = 13
        return i
    
    def __repr__(self):
        return f"{self.rank}{self.suit}"
    
    @property
    def sort_key(self):
        return (self.suit, self.rank_index)
    
_deck = [
        SergeantMajorCard(suit=suit, rank=rank) 
        for suit in Card.valid_suit[:4] 
        for rank in Card.valid_rank
    ]

