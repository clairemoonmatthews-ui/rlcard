'''
    File name: sergeant_major/player.py
    Author: Claire Matthews
    Date created: 10/9/2025
'''

from typing import List

from rlcard.games.base import Card

from .utils import utils


class SergeantMajorPlayer:

    def __init__(self, player_id: int, np_random):
        ''' Initialize a SergeantMajor player class

        Args:
            player_id (int): id for the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.hand = []  # type: List[Card]
    
    def get_player_id(self) -> int:
        ''' Return player's id
        '''
        return self.player_id


    def did_populate_hand(self):
        pass

    def add_card_to_hand(self, card: Card):
        self.hand.append(card)

    def remove_card_from_hand(self, card: Card):
        self.hand.remove(card)

    def __str__(self):
        return str(self.player_id)

    @staticmethod
    def short_name_of(player_id: int) -> str:
        return str(player_id) 

    

    