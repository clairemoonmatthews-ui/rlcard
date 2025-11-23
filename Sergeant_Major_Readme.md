What is SM? 
What is simplified SM?
What is RLCARD?
What did I create? Environment, game, heuristic agent
What experiments have I run? DQN (small boost), HeuristicAgent (large boost, win rate 56-57% in 3-player game)
Next steps? ML agent, sequence based

# Claire Matthews Math/CS Senior Thesis Progress Check
# Reinforcement Learning On The Card Game Sergeant Major
# November 23, 2025

## Game Rules
Sergeant major is an asymmetrical trick taking card game between three players. 
To win the game a player must win at least twelve tricks. 
The game is broken up into rounds. 
Each player is dealt sixteen cards, and the four remaining cards are put to the side in a pile called the kitty. 
In each round, each player is assigned a different goal for how many tricks they are trying to reach. 
Players are always trying to reach the winning twelve tricks, but if they do not at least make their goal tricks then there will be a penalty. 
The dealer's goal is 8 tricks, the left of the dealer is 5, and the right of the dealer is 3. The players with the higher goals receive slight benefits to make up for the fact their goals will be more difficult to reach. The dealer gets to select the trump suit. The trump suit will be the most powerful suit for that round. If a card of that suit is played during any trick it will win over any other suit. After the dealer selects the trump suit they select four cards from their hand to swap with the cards from the kitty pile. The player to the left of the dealer gets to start the round by playing the first card that marks the start of the first trick. Each trick will be made up of three cards, one played by each player. The winner of the previous trick will start the next trick. The suit of the first card played is the leading suit. For each of the following players if they are able to play a card of the suit led, they must. The trick is one by the player with the highest trump card played, or if no trump card is played, the highest card of the leading suit. Aces are high. At the end of the round the position of the dealer moves to the right. For each trick that a player made above their goal, they get to select any card from their hand the following round, to give to the players that did not reach their goal. The players that did not reach their goal then must take the highest valued card from their hand, of the suit of the card they received, to return back to the player. This occurs after the kitty swap. The game continues on like this, rotating dealers, until a player has won.

### Simplified version
Simplified Sergeant Major is a single hand version of the game. There is no kitty, no swapping of cards, and the trump suit is selcted randomly. The game is won by the player with the most tricks.

## Code 
I have written code for the environment of Simplified Sergeant Major.
I hope to support the full game later.

I am using a library called RLCARD that has resources to run reinforcement learning on card games. The library had games such as Rummy and Bridge, which I used to model my own environment after. 

I created a random agent for the game that selects legal actions at random. The first trial I did was I ran the DQN agent from the RLCARD library against my random agent. The DQN did not perform as well as I had hoped. It was only winning over the random agents by a slight margin. This inspired me to create a heuristic agent for my own strategy in the game. I am using this agent as a baseline sanity check.  I ran a tournament between my own agent and two random agents. The heuristic agent wins 56-57% of games, which is really good for a three-player game, a raise of 23pp over the baseline of 33%. 

## The Next Steps: 

* Get a reinforcement learning agent to perform as well, if not better than my own hand-coded heuristic agent.
* Expand the environment to the full game.
* Do more research into which reinforcement learning agents I should be using. I think that using an agent that does well with sequences may produce better results.
* Train an agent that can consistently beat my heuristic agent.
* Analyze the reinforcement learning agents strategy and improve my own game play.
* Analyze my results

