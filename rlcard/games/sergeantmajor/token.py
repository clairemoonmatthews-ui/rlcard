from enum import IntEnum
from typing import Optional

from rlcard.games.base import Card
from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.types import PlayerId


max_state_length = 163
vocab_size = 63
actions = 52

# 0-51 card, 52-54 player, 55-58 trump, 59 start_trick, 60 winner, 61 hand, 62 end
# e.g. hand, player1, card_6S, ...x16..., trump_S, start_trick, player1, card_6S, player2, card_KH, player3, card_7S, winner, player3, ...x16... end

class Token(IntEnum):
    FIRST_CARD = 0
    FIRST_PLAYER = 52
    FIRST_SUIT = 55
    START_TRICK = 59
    WINNER = 60
    HAND = 61
    END = 62
    
    @classmethod
    def card(cls, token: int) -> Optional[SergeantMajorCard]:
        """If token represents a card, returns the card; otherwise returns None"""
        if token in range(cls.FIRST_CARD, cls.FIRST_PLAYER):
            return SergeantMajorCard.from_index(token - cls.FIRST_CARD)
        else:
            return None

    @classmethod
    def player(cls, token: int) -> Optional[PlayerId]:
        """If token represents a player, returns the player id; otherwise returns None"""
        if token in range(cls.FIRST_PLAYER, cls.FIRST_SUIT):
            return token.item() - cls.FIRST_PLAYER
        else:
            return None

    @classmethod
    def suit(cls, token: int) -> Optional[str]:
        """If token represents a (trump) suit, returns the suit string (e.g. "S"); otherwise returns None"""
        if token in range(cls.FIRST_SUIT, cls.START_TRICK):
            return Card.valid_suit[token - cls.FIRST_SUIT]
        else:
            return None
        
    
    

