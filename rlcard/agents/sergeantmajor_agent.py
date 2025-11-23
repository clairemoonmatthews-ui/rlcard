from dataclasses import dataclass
import random
from typing import Counter, Dict, List, Optional, Set, Tuple

import numpy as np

from rlcard.games.sergeantmajor.card import SergeantMajorCard
from rlcard.games.sergeantmajor.types import Hand, PlayerId, Trick, Tricks
from rlcard.envs.sergeantmajor import SergeantMajorEnv

import logging

logger = logging.getLogger(__name__)

Token = SergeantMajorEnv.Token
Suit = str
Cards = List[SergeantMajorCard]
Card = SergeantMajorCard

class HeuristicAgent:
    """
    Heuristic agent for Simplified Sergeant Major.
    Implements rule-based strategy for trick-taking.
    """

    @dataclass
    class State:
        hand: Hand
        trump_suit: Suit
        current_trick: Trick
        trick_history: Tricks
        player: PlayerId
        subsequent_players: Set[PlayerId]
    
    def __init__(self, num_actions=52):
        self.use_raw = False  # RLCard convention
        self.num_actions = num_actions
        
    def step(self, state:Dict) -> int:
        """
        Called during gameplay to select an action.
        
        Args:
            state: Dict with 'obs' (token sequence) and 'legal_actions' (list of valid card indices)
        
        Returns:
            action: Integer in range [0, 51] representing card to play
        """
        legal_actions = state['legal_actions']
        obs = state['obs']  # The token sequence
        
        # Parse the observation to extract game state
        parsed_state = self._parse_state(obs)
        logger.info(parsed_state)
        parsed_legal_actions: List[SergeantMajorCard] = [ Card.from_index(action) for action in legal_actions ]
        
        # Apply heuristic rules
        card = self._choose_action(parsed_state, parsed_legal_actions)
        # print(card)
        return card.get_index()
    
    def eval_step(self, state:Dict) -> Tuple[int, Dict]:
        """
        Called during evaluation - returns action and info dict.
        
        Args:
            state: Same as step()
        
        Returns:
            action: Selected action
            info: Dict with any debugging/analysis info (can be empty)
        """
        action = self.step(state)
        return action, {}
    
    def _parse_state(self, obs: np.array) -> State:
        """
        Parse token sequence into useful game state components.
        
        Args:
            obs: Token sequence (list or array of token IDs)
        
        Returns:
            parsed_state: 
                hand: List of card tokens in your hand
                trump_suit: Trump suit token (or None)
                current_trick: List of (player, card) tuples for current trick
                trick_history: List of completed tricks
        """
        # print(obs)
        hand = []
        trump_suit = None
        current_trick = []
        trick_history = []
        subsequent_players = set(range(3)) 
        i = 0 
        # read hand
        while i < len(obs) and (trump_suit := Token.suit(obs[i])) is None:
            token = obs[i]
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
        subsequent_players.remove(player)
        for p, _ in current_trick:
            subsequent_players.discard(p)
        state = self.State(hand=hand, trump_suit=trump_suit, current_trick=current_trick, trick_history=trick_history, player=player, subsequent_players=subsequent_players)
        return state

    
    def _choose_action(self, state: State, legal_actions: Cards) -> Card:
        """
        Apply heuristic rules to choose an action.
        
        Args:
            state: hand, tricks, etc.
            legal_actions: Valid cards you can play
        
        Returns:
            action: Card index to play
        """

        def _card_rank(card: Card, suit_led: str) -> int:
            card_rank = Card.valid_rank.index(card.rank) + 1
            card_rank = card_rank if card_rank > 1 else 14
            if card.suit == state.trump_suit:
                card_rank += 13 
            elif card.suit != suit_led:
                card_rank = 0
            return card_rank

        def determine_winning_card() -> Optional[Card]:
                """
                Determine which card is winning the current trick.
                The winner is the highest card of the lead suit
                (or highest trump if trump suit is implemented).
                
                Returns:
                    The card the is currently winning the trick
                """
    
                if not state.current_trick:
                    return None
                trick = state.current_trick
                result = trick[0][1]
                suit_led = trick[0][1].suit
                best_rank = _card_rank(trick[0][1], suit_led)
                for _, card in trick[1:]:
                    rank = _card_rank(card, suit_led)
                    if rank > best_rank:
                        best_rank = rank
                        result = card
                return result

        def filter_winners(cards: Cards) -> Cards:
            """Returns only cards that would be the winners of the current trick"""
            if not state.current_trick:
                return cards
            result = []
            suit_led = state.current_trick[0][1].suit
            winning_current_trick = determine_winning_card()
            winning_current_trick_rank = _card_rank(winning_current_trick, suit_led)
            for card in cards:
                rank = _card_rank(card, winning_current_trick.suit)
                if rank > winning_current_trick_rank:
                    result.append(card)
            return result

        def filter_declared(cards: Cards) -> Cards:
            """Returns only cards where no subsequent player might have a higher one"""
            result = []
            missing = missing_cards()
            for card in cards:
                missing_in_suit = filter_by_suit(missing, card.suit)
                if not missing_in_suit:
                    result.append(card)
                    continue
                max_missing = max_rank(missing_in_suit)
                if card.rank_index > max_missing.rank_index:
                    result.append(card)
            return result

        def filter_declared_trump(cards: Cards) -> Cards:
            """Returns only cards that are trump where no subsequent player might have a higher one"""
            return filter_declared(filter_by_suit(cards, state.trump_suit))

        def played() -> Cards:
            """Returns the set of cards already played"""
            result = []
            for trick in state.trick_history:
                for _, card in trick:
                    result.append(card)
            for _, card in state.current_trick:
                result.append(card)
            return result
        
    
        def missing_cards() -> Cards:
            """Returns the set of cards that are neither played, nor in the player's hand."""
            deck = set(Card.get_deck())
            result = list(deck.difference(state.hand).difference(played()))
            # print(f"{deck=}, {state.hand=}, {played()=}, {result=}")
            return result
    

        def filter_by_suit(cards: Cards, suit: Suit) -> Cards:
            """Returns the subset of cards in some suit"""
            result = []
            for card in cards:
                if card.suit == suit:
                    result.append(card)
            return result
        
        def n_trumps_out() -> int:
            """Returns the number of trumps that are neither played nor in the current player's hand (or for dealer, in the kitty)"""
            return len(filter_by_suit(missing_cards(), state.trump_suit))

        def shortest_suit(cards: Cards) -> Suit:
            """Returns the suit with the fewest cards in the set; if there is a tie, the lower pip count is selected, or the choice is arbitrary.  Empty suits will not be considered."""
            suit_counts = Counter()
            for card in cards:
                suit_counts[card.suit] += 1
            suit_counts = suit_counts.most_common()
            return suit_counts[-1][0]            

        def known_voids() -> Set[Suit]:
            """Returns a set of known suit voids for the subsequent players
            """
            result = set()
            for trick in state.trick_history:
                suit_led = trick[0][1].suit
                for p, card in trick[1:]:
                    if card.suit != suit_led and p in state.subsequent_players:
                        result.add(suit_led)
            return result

        def filter_does_not_play_into_void(cards: Cards) -> Cards:
            """Returns cards that no player is known to be void in, includes all trump cards
            """
            voids = known_voids()
            return [c for c in cards 
                    if c.suit not in voids or c.suit == state.trump_suit]

        def min_rank(cards: Card) -> Card:
            """Returns the card with minimum rank"""
            return min(cards, key = lambda c: c.rank_index)
        
        def max_rank(cards: Card) -> Card:
            """Returns the card with maximum rank"""
            return max(cards, key = lambda c: c.rank_index)

        def choose(cards: Cards) -> Card:
            """Chooses a random card"""
            return random.choice(cards)

        def sort_by_rank(cards: Cards) -> Cards:
            """ Sorts cards by increasing rank"""
            return sorted(cards, key = lambda c: c.rank_index)

        if not state.current_trick: # first player
            # if you have declared trumps, play it
            if cards := filter_declared_trump(legal_actions):
                logger.info("First player: Playing declared trump")
                return choose(cards)
            # if you have declared card with no known players void, play it 
            elif cards := filter_declared(filter_does_not_play_into_void(legal_actions)):
                logger.info("First player: Playing declared not into void")
                return choose(cards)
            # if you have declared card with players void but no trumps, play it
            elif (cards := filter_declared(legal_actions)) and n_trumps_out() == 0:
                logger.info("First player: Playing declared with no trumps out")
                return choose(cards)
            else: # play the lowest card of the suit they have the least of 
                suit = shortest_suit(legal_actions)
                logger.info("First player: Playing lowest from shortest suit")
                return min_rank(filter_by_suit(legal_actions, suit))

        elif len(state.current_trick) == 1: # second player
            if winners := filter_does_not_play_into_void(filter_winners(legal_actions)):
                if declared := filter_declared(winners):
                    logger.info("Second player: Playing declared winner not into void")
                    return min_rank(declared)
                else:
                    logger.info("Second player: Playing minimum rank winner")
                    return min_rank(winners)
            else:
                logger.info("Second player: Playing minimum rank")
                return min_rank(legal_actions)
                # suit = shortest_suit(legal_actions)
                # logger.info("Second player: Playing minimum rank from shortest suit")
                # return min_rank(filter_by_suit(legal_actions, suit))
            
        else: # third player
            if winners := filter_winners(legal_actions):
                logger.info("Third player: Playing minimum rank winner")
                return min_rank(winners)
            else:
                logger.info("Third player: Playing minimum rank")
                return min_rank(legal_actions)

        