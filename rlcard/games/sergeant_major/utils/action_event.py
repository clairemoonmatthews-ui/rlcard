'''
    File name: sergeant_major/action_event.py
    Author: Claire Matthews
    Date created: 10/9/2025
'''

from rlcard.games.base import Card

from . import utils as utils

# ====================================
# Action_ids:
#        0 to 2 -> score_player_id
#        3 to 54 -> play_card_id
# ====================================

n_players = 3
score_player_action_id = 0
play_card_action_id = score_player_action_id + n_players


class ActionEvent(object):

    def __init__(self, action_id: int):
        self.action_id = action_id

    def __eq__(self, other):
        result = False
        if isinstance(other, ActionEvent):
            result = self.action_id == other.action_id
        return result

    @staticmethod
    def get_num_actions():
        ''' Return the number of possible actions in the game
        '''
        return play_card_action_id + 52

    @staticmethod
    def decode_action(action_id) -> 'ActionEvent':
        ''' Action id -> the action_event in the game.

        Args:
            action_id (int): the id of the action

        Returns:
            action (ActionEvent): the action that will be passed to the game engine.
        '''
        if action_id in range(score_player_action_id, score_player_action_id + n_players):
            player_id = action_id - score_player_action_id
            action_event = ScorePlayerAction(player_id)
        elif action_id in range(play_card_action_id, play_card_action_id + 52):
            card_id = action_id - play_card_action_id
            card = utils.get_card(card_id=card_id)
            action_event = PlayCardAction(card=card)
        else:
            raise Exception("decode_action: unknown action_id={}".format(action_id))
        return action_event


class ScorePlayerAction(ActionEvent):

    def __init__(self, player_id):
        super().__init__(action_id=score_player_action_id + player_id)
        self.player = player_id

    def __str__(self):
        return "Score {}".format(str(self.player))


class PlayCardAction(ActionEvent):

    def __init__(self, card: Card):
        card_id = utils.get_card_id(card)
        super().__init__(action_id=play_card_action_id + card_id)
        self.card = card

    def __str__(self):
        return "play {}".format(str(self.card))

