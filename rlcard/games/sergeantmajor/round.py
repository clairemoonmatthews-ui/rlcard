import random
from typing import List, Tuple, Optional
import numpy as np
from rlcard.agents.sergeantmajor_agent import Cards
from rlcard.games.base import Card
from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.types import PlayerId, Trick, Tricks
from rlcard.utils.seeding import np_random
from rlcard.games.sergeantmajor.state import PlayerState


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
        self.trump_suit = Card.valid_suit[self.np_random.choice(range(4))]


    def proceed_round(self, action: Card) -> None:
        """
        Process one card play and advance the game state.
        Handles trick completion and round completion.
        
        Args:
            action: The card being played by the current player
        """
        assert action in self.hands[self.current_player_id], f"action = {action} hand = {self.hands[self.current_player_id]} "
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

    @classmethod
    def from_player_state(cls, state:PlayerState, np_random: np.random.RandomState = None):
        if np_random is None:
            np_random = np.random.RandomState()
        num_players = 3
        round = cls(np_random, num_players)
        round.current_player_id = state.current_player
        round.tricks = state.tricks
        round.trump_suit = state.trump_suit
        round.winners = [round._determine_trick_winner(trick) for trick in round.tricks]
        round.won_trick_counts = [round.winners.count(i) for i in range(num_players)]
        round.current_trick = state.current_trick
        round.hands = [[] for i in range(num_players)]
        round.hands[round.current_player_id] = state.hand
        round._determinize_hands()
        return round
        
    def _determinize_hands(self):

        def played() -> Cards:
            """Returns the set of cards already played"""
            result = []
            for trick in self.tricks:
                for _, card in trick:
                    result.append(card)
            for _, card in self.current_trick:
                result.append(card)
            return result
        
        def missing_cards():
            """Returns the set of cards that are neither played, nor in the player's hand."""
            deck = set(SergeantMajorCard.get_deck())
            result = deck.difference(self.hands[self.current_player_id]).difference(played())
            # print(f"{deck=}, {state.hand=}, {played()=}, {result=}")
            return result
        
        def played_in_current_trick(player_id) -> bool:
            result = False
            for i,_ in self.current_trick:
                if player_id == i:
                    result = True
            return result
        

        
        def cards_needed():
            num_needed = [16 - len(self.hands[i]) for i in range(self.num_players)]
            for trick in self.tricks + [self.current_trick]:
                for p, _ in trick:
                    num_needed[p] -= 1
            return num_needed
        
        def known_voids():
            result = [set() for _ in range(self.num_players)]
            for trick in self.tricks + [self.current_trick]:
                if len(trick) != 0:
                    suit_led = trick[0][1].suit
                    for p, c in trick[1:]:
                        if c.suit != suit_led:
                            result[p].add(suit_led)
            return result
        
        def potential_cards():
            missing = missing_cards()
            return [[c for c in missing if c.suit not in voids[i]] for i in range(self.num_players)]
        
        # Generate legal determinization
        # 1. Calculate how many cards each player needs
        # 2. Determine each player's known voids from history
        # 3. Determine potential cards for each player
        # 4. For player with fewer "spare" cards:
        # 4a. Select cards from potential cards
        # 4b. Add to this player's hand
        # 4c. Remove from other player's potential cards
        # 5. For remaining player, select and add cards

        
        c_needed = cards_needed()
        voids = known_voids()
        p_cards = potential_cards()
        players = sorted(range(self.num_players), key=lambda p: len(p_cards[p]) - c_needed[p])
        cards_dealt = set()
        for p in players:
            cards = set(p_cards[p]) - cards_dealt
            chosen = self.np_random.choice(list(p_cards[p]), c_needed[p], replace=False)
            self.hands[p].extend(chosen)
            cards_dealt.update(chosen)


                
        
