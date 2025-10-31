import random
from typing import List, Tuple, Optional
import numpy as np
from rlcard.games.base import Card
from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.types import PlayerId, PlayerState, Trick, Tricks
from rlcard.utils.seeding import np_random


class SergeantMajorRound:
    """
    Manages the state and logic for one complete round (deal) of Sergeant Major.
    Contains all the core game mechanics.
    """
    
    def __init__(self, np_random: np.random.RandomState, num_players: int) -> None:
        """
        Initialize a new round of Sergeant Major.
        
        Args:
            np_random: Random number generator for shuffling
            num_players: Number of players (should be 3)
        """
        self.np_random = np_random
        self.current_player_id: PlayerId = 0
        self.won_trick_counts = [0]*num_players
        self.tricks: Tricks = []
        self.winners: List[PlayerId ]= []
        self.current_trick: Trick = []
        self.num_players = num_players
        self._deal_cards() # Adds self.hands, self.trump_suit
    
    def _deal_cards(self) -> None:
        """
        Shuffle the deck and deal cards to all players.
        In standard Sergeant Major, deal 16 cards to each of 3 players.
        """
        initial_hand_size = 16
        deck = SergeantMajorCard.get_deck() 
        self.np_random.shuffle(deck)
        self.hands = []
        for p in range(self.num_players):
            self.hands.append(deck[p * initial_hand_size:(1+p) * initial_hand_size]) 
            
        # four cards left in the deck that we are ignoring for now

        # in simplified sergeant major the trump suit is selected at random
        self.trump_suit = self.np_random.choice(Card.valid_suit[:4])

    def proceed_round(self, action: Card) -> None:
        """
        Process one card play and advance the game state.
        Handles trick completion and round completion.
        
        Args:
            action: The card being played by the current player
        """
        self.current_trick.append((self.current_player_id, action))
        self.hands[self.current_player_id].remove(action)
        if self._is_current_trick_complete():
            self.tricks.append(self.current_trick)
            winner = self._determine_trick_winner(self.current_trick)
            self.winners.append(winner)
            self.won_trick_counts[winner] += 1
            self.current_trick = []
            self.current_player_id = winner
        else:
            self.current_player_id = (self.current_player_id + 1) % self.num_players

    def _is_current_trick_complete(self) -> bool:
        return len(self.current_trick) == self.num_players
    
    def _determine_trick_winner(self, trick) -> int:
        """
        Determine which player won the current trick.
        The winner is the player who played the highest card of the lead suit
        (or highest trump if trump suit is implemented).
        
        Returns:
            Player ID of the trick winner
        """
        winner = trick[0][0]
        suit_led = trick[0][1].suit
        best_rank = self._card_rank(trick[0][1], suit_led)
        for player, card in trick[1:]:
            rank = self._card_rank(card, suit_led)
            if rank > best_rank:
                best_rank = rank
                winner = player
        return winner
    
    def _card_rank(self, card: Card, suit_led: str) -> int:
        """
        Convert a card's rank to a numeric value for comparison.
        
        Args:
            card: The card to evaluate
            
        Returns:
            Numeric rank value (2-14, where Ace=14) for the card of the suit led
            adds 13 to the numeric rank value of the the trump suit 
            and returns zero for all other cards
        """
        card_rank = Card.valid_rank.index(card.rank) + 1
        card_rank = card_rank if card_rank > 1 else 14
        if card.suit == self.trump_suit:
            card_rank += 13 
        elif card.suit != suit_led:
            card_rank = 0
        return card_rank
    
    def get_legal_actions(self, player_id: int) -> List[Card]:
        """
        Get the list of legal cards the player can play.
        Must follow suit if possible.
        
        Args:
            player_id: The player whose legal actions to get
            
        Returns:
            List of Card objects that are legal to play
        """
        hand = self.hands[player_id]
        if self.current_trick: 
            suit_led = self.current_trick[0][1].suit
            result = [card for card in hand if suit_led == card.suit]
            if result:
                return result
        return hand
    
    def get_state(self, player_id: int) -> PlayerState:
        """
        Get the observable state for a specific player.
        
        Args:
            player_id: The player whose perspective to get
            
        Returns:
            Dictionary containing:
                - hand: List of cards in player's hand
                - current_trick: Cards played so far this trick
                - tricks_won: Number of tricks each player has won
                - legal_actions: Cards this player can legally play
                - current_player: ID of player whose turn it is
        """
        return PlayerState(current_player=player_id, hand=self.hands[player_id], current_trick=self.current_trick, tricks=self.tricks, tricks_won=self.won_trick_counts[player_id], legal_actions=self.get_legal_actions(player_id), trump_suit=self.trump_suit, winners=self.winners)
        
        