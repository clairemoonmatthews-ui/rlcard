from collections import OrderedDict
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional
import numpy as np
from rlcard.envs import Env
from rlcard.games.base import Card
from rlcard.games.sergeantmajor import SergeantMajorGame
from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.judger import SergeantMajorJudger
from rlcard.games.sergeantmajor.types import PlayerId, PlayerState, Trick

class SergeantMajorEnv(Env):
    """
    Sergeant Major Environment wrapper for RLCard.
    Provides the standardized environment interface for agents to interact with the game.
    """
    
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



    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize a Sergeant Major environment.
        
        Args:
            config: Configuration dictionary containing environment settings
        """
        self.game = SergeantMajorGame()
        self.name = "sergeant-major"
        super().__init__(config)
        self.state_shape = [[self.max_state_length]] # maximum length
        self.action_shape = [None]
        self.pad_state = config.get("sergeant-major.pad_state", False)
    
    def _extract_state(self, state: PlayerState) -> Dict[str, Any]:
        """
        Extract and encode the state for the agent.
        Converts the raw game state into the format expected by agents.
        
        Args:
            state: Raw state from the game
            
        Returns:
            Processed state dictionary with 'obs' and 'legal_actions' keys
        """
        # 0-51 card, 52-54 player, 55-58 trump, 59 start_trick, 60 winner, 61 hand, 62 end
        # e.g. hand, player1, card_6s, ...x16..., trump_spades, start_trick, player1, card_6s, player2, card_kh, player3, card_7s, winner, player3, ...x16... end
        def emit(token: int, name: str):
            obs.append(token)
            raw_obs.append(name)
        def emit_player(player_id: PlayerId):
            emit(self.Token.FIRST_PLAYER + player_id, f"player_{player_id + 1}")
        def emit_card(card: Card):
            emit(card.get_index(), f"card_{card}")
        def emit_trick(trick: Trick):
            emit(self.Token.START_TRICK, "start_trick")
            for player_id, card in trick:
                emit_player(player_id)
                emit_card(card)
                
        obs = []
        raw_obs = []
        emit(self.Token.HAND, "hand")
        emit_player(state.current_player)
        for card in state.hand:
            emit_card(card)
        emit(self.Token.FIRST_SUIT + state.trump_suit, f"trump_{Card.valid_suit[state.trump_suit]}")
        for trick, winner in zip(state.tricks, state.winners):
            emit_trick(trick)
            emit(self.Token.WINNER, "winner")
            emit_player(winner)
        if self.game.is_over():
            emit(self.Token.END, "end")
        else: 
            emit_trick(state.current_trick)
            emit_player(state.current_player)
        assert len(obs) <= self.max_state_length
        obs = np.array(obs)
        if self.pad_state:
            obs = np.pad(obs, (0,self.max_state_length - len(obs)), constant_values = self.Token.END)
        assert len(obs) <= self.max_state_length, len(obs)
        
        legal_actions = {}
        raw_legal_actions = []
        for action in state.legal_actions:
            legal_actions[(action.get_index())] = None
            raw_legal_actions.append(str(action))
        legal_actions = OrderedDict(legal_actions)

        extracted_state = {
            'obs': obs, 
            'raw_obs': raw_obs,
            'legal_actions': legal_actions, 'raw_legal_actions': raw_legal_actions}
        return extracted_state
    
    def _decode_action(self, action_id: int) -> Any:
        """
        Decode an action ID into the game's action format.
        Converts integer action indices to Card objects or other game actions.
        
        Args:
            action_id: Integer action ID from the agent
            
        Returns:
            Game-specific action (typically a Card object)
        """
        return SergeantMajorCard.from_index(action_id)
    
    def _get_legal_actions(self) -> List[int]:
        """
        Get the list of legal action IDs for the current player.
        
        Returns:
            List of integer action IDs that are currently legal
        """
        legal_actions_cards = self.game.round.get_legal_actions(self.game.get_player_id())
        legal_actions_int = [c.get_index() for c in legal_actions_cards]
        return legal_actions_int
    
    def get_payoffs(self) -> List[float]:
        """
        Get the final payoffs for all players.
        
        Returns:
            List of payoffs (one per player)
        """
        return np.array(SergeantMajorJudger.judge_game(self.game))
    
    def _load_model(self) -> None:
        """
        Load any pretrained models if specified in config.
        Optional method for loading saved models.
        """
        pass