# chess-behaviour

## Abstract

In recent years, machine learning models have been able to match or surpass human abilities in
some specific tasks, such as image recognition. In chess, we can recognize that different players
have different playing styles. This gives rise to the question: “Can we predict who is playing a
chess game from their moves alone?”.
In this project, we develop a deep neural network that tackles this task. We use a convolutional-
LSTM model that learns patterns in expert chess player decisions with the goal of predicting
who the player is. We use the few-shot classification framework to generalize our model to new
players with few reference examples. Our model achieves 13.4% results when distinguishing
between the 400 players it was trained on. This is 53 times better than guessing randomly. The
model was also 51 times more accurate than a random guesser on previously unseen players
when using only 50 games from each of them as a reference.
