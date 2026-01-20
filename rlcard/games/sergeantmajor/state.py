from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from rlcard.agents.sergeantmajor_agent import Card, Suit
from rlcard.games.sergeantmajor import token
from rlcard.games.sergeantmajor.types import Actions, Hand, PlayerId, Trick, Tricks
from rlcard.games.sergeantmajor.token import Token


@dataclass
class PlayerState:
    """Type definition for a player's observable self."""
    current_player: PlayerId
    hand: Hand 
    current_trick: Trick 
    tricks: Tricks 
    tricks_won: int 
    legal_actions: Actions 
    trump_suit: Suit
    winners: List[PlayerId]

    def to_tokens(self, padding=False) -> Dict[str, Any]:
        """
        Extract and encode the self for the agent.
        Converts the raw game self into the format expected by agents.
        
        Args:
            self: Raw self from the game
            
        Returns:
            Processed self dictionary with 'obs' and 'legal_actions' keys
        """
        # 0-51 card, 52-54 player, 55-58 trump, 59 start_trick, 60 winner, 61 hand, 62 end
        # e.g. hand, player1, card_6s, ...x16..., trump_spades, start_trick, player1, card_6s, player2, card_kh, player3, card_7s, winner, player3, ...x16... end
        def emit(token: int, name: str):
            obs.append(token)
            raw_obs.append(name)
        def emit_player(player_id: PlayerId):
            emit(Token.FIRST_PLAYER + player_id, f"player_{player_id + 1}")
        def emit_card(card: Card):
            emit(card.get_index(), f"card_{card}")
        def emit_trick(trick: Trick):
            emit(Token.START_TRICK, "start_trick")
            for player_id, card in trick:
                emit_player(player_id)
                emit_card(card)
                
        obs = []
        raw_obs = []
        emit(Token.HAND, "hand")
        emit_player(self.current_player)
        for card in self.hand:
            emit_card(card)
        emit(Token.FIRST_SUIT + self.trump_suit, f"trump_{Card.valid_suit[self.trump_suit]}")
        for trick, winner in zip(self.tricks, self.winners):
            emit_trick(trick)
            emit(Token.WINNER, "winner")
            emit_player(winner)
        if self.game.is_over():
            emit(Token.END, "end")
        else: 
            emit_trick(self.current_trick)
            emit_player(self.current_player)
        assert len(obs) <= self.max_self_length
        obs = np.array(obs)
        if padding:
            obs = np.pad(obs, (0,token.max_self_length - len(obs)), constant_values = Token.END)
        assert len(obs) <= token.max_self_length, len(obs)
        return obs, raw_obs 
        
    def is_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game has ended, False otherwise
        """
        return len(self.tricks) == 16
    
    @classmethod
    def from_tokens(cls, obs:np.array) -> "PlayerState":
        """
        Parse token sequence into useful game state components.
        
        Args:
            obs: Token sequence (list or array of token IDs)
        
        Returns: new player state
        """
        # print(obs)
        hand = []
        trump_suit = None
        current_trick = []
        trick_history = []
        i = 0 
        # read hand
        while i < len(obs) and (trump_suit := Token.suit(obs[i])) is None:
            token = obs[i]:
            if (card := Token.card(token)) is not None:
                hand.append(card)
            i += 1
        hand = sorted(hand, key=lambda c: c.sort_key)
        # print(f"{hand=}, {i=}, {trump_suit=}")
        # read tricks
        while i < len(obs):
            token = obs[i]
            if token == Token.START_TRICK:
                if current_trick:
                    trick_history.append(current_trick)
                    current_trick = []
            elif (player_candidate := Token.player(token)) is not None:
                player = player_candidate
            elif (card := Token.card(token)) is not None:
                assert player is not None, f"no player, trying to play {card=}, {token=}"
                current_trick.append((player, card))
            i += 1
        state = cls(hand=hand, trump_suit=trump_suit, current_trick=current_trick, tricks=trick_history, current_player=player)
        return state
    
    
    
    
    tricks_won: int 
    legal_actions: Actions 
    
    winners: List[PlayerId]