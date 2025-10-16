'''
    File name: sergeant_major/move.py
    Author: Claire Matthews
    Date created: 10/9/2025
'''

from typing import List

from ..player import SergeantMajorPlayer

from .action_event import *

from .sergeant_major_error import SergeantMajorProgramError


#
#   These classes are used to keep a move_sheet history of the moves in a round.
#

class SergeantMajorMove(object):
    pass


class PlayerMove(SergeantMajorMove):

    def __init__(self, player: SergeantMajorPlayer, action: ActionEvent):
        super().__init__()
        self.player = player
        self.action = action


class DealHandMove(SergeantMajorMove):

    def __init__(self, player_dealing: SergeantMajorPlayer, shuffled_deck: List[Card]):
        super().__init__()
        self.player_dealing = player_dealing
        self.shuffled_deck = shuffled_deck

    def __str__(self):
        shuffled_deck_text = " ".join([str(card) for card in self.shuffled_deck])
        return "{} deal shuffled_deck=[{}]".format(self.player_dealing, shuffled_deck_text)


class PlayCardMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: PlayCardAction):
        super().__init__(player, action)
        if not isinstance(action, PlayCardAction):
            raise SergeantMajorProgramError("action must be PlayCardAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)


class ScorePlayerMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer,
                 action: ScorePlayerAction):
        super().__init__(player, action)
        if not isinstance(action, ScorePlayerAction):
            raise SergeantMajorProgramError("action must be ScorePlayerAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)
