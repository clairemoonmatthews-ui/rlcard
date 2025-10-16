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


class DrawCardMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: DrawCardAction, card: Card):
        super().__init__(player, action)
        if not isinstance(action, DrawCardAction):
            raise SergeantMajorProgramError("action must be DrawCardAction.")
        self.card = card

    def __str__(self):
        return "{} {} {}".format(self.player, self.action, str(self.card))


class PickupDiscardMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: PickUpDiscardAction, card: Card):
        super().__init__(player, action)
        if not isinstance(action, PickUpDiscardAction):
            raise SergeantMajorProgramError("action must be PickUpDiscardAction.")
        self.card = card

    def __str__(self):
        return "{} {} {}".format(self.player, self.action, str(self.card))


class DeclareDeadHandMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: DeclareDeadHandAction):
        super().__init__(player, action)
        if not isinstance(action, DeclareDeadHandAction):
            raise SergeantMajorProgramError("action must be DeclareDeadHandAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)


class DiscardMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: DiscardAction):
        super().__init__(player, action)
        if not isinstance(action, DiscardAction):
            raise SergeantMajorProgramError("action must be DiscardAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)


class KnockMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: KnockAction):
        super().__init__(player, action)
        if not isinstance(action, KnockAction):
            raise SergeantMajorProgramError("action must be KnockAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)


class GinMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer, action: GinAction):
        super().__init__(player, action)
        if not isinstance(action, GinAction):
            raise SergeantMajorProgramError("action must be GinAction.")

    def __str__(self):
        return "{} {}".format(self.player, self.action)


class ScoreNorthMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer,
                 action: ScoreNorthPlayerAction,
                 best_meld_cluster: List[List[Card]],
                 deadwood_count: int):
        super().__init__(player, action)
        if not isinstance(action, ScoreNorthPlayerAction):
            raise SergeantMajorProgramError("action must be ScoreNorthPlayerAction.")
        self.best_meld_cluster = best_meld_cluster  # for information use only
        self.deadwood_count = deadwood_count  # for information use only

    def __str__(self):
        best_meld_cluster_str = [[str(card) for card in meld_pile] for meld_pile in self.best_meld_cluster]
        best_meld_cluster_text = "{}".format(best_meld_cluster_str)
        return "{} {} {} {}".format(self.player, self.action, self.deadwood_count, best_meld_cluster_text)


class ScoreSouthMove(PlayerMove):

    def __init__(self, player: SergeantMajorPlayer,
                 action: ScoreSouthPlayerAction,
                 best_meld_cluster: List[List[Card]],
                 deadwood_count: int):
        super().__init__(player, action)
        if not isinstance(action, ScoreSouthPlayerAction):
            raise SergeantMajorProgramError("action must be ScoreSouthPlayerAction.")
        self.best_meld_cluster = best_meld_cluster  # for information use only
        self.deadwood_count = deadwood_count  # for information use only

    def __str__(self):
        best_meld_cluster_str = [[str(card) for card in meld_pile] for meld_pile in self.best_meld_cluster]
        best_meld_cluster_text = "{}".format(best_meld_cluster_str)
        return "{} {} {} {}".format(self.player, self.action, self.deadwood_count, best_meld_cluster_text)
