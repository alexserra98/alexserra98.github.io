---
title: "Deep Q-Network for ATARI Snake Game"
date: 2023-09-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/projects/Deep_Q-Network_for_ATARI_Snake_Game/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
## Project: Deep Q-Network for ATARI Snake Game

### Overview

The aim of the project was to implement Deep QNetworks, following the description contained in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and in particular to exploit it to make an agent learn how to play Snake game.

Instead of directly implementing DQN using CNNs, we progressively build up different models, starting from basic Qlearning algorithms and moving towards MLP, in order to deal with increasing complexity of the game. 
The code for all the trials we made is available, along with one simple version of Snake game, which can be played using all the models trained, from standard tabular methods to DQN.

- **Key Technologies**: Python, Pytorch, OpenAI Gym
{{<figure src="images/snake.gif" caption="The agent finally learns to play the game" width="400" height="400">}}<br>
### Project Details
This worked serves as unified enviroment to train and test different models on the Snake game. The game is implemented using OpenAI Gym, which provides a simple interface to interact with the game. The code is structured in the following way:
- *game* folder: it contains the code to load a pretrained model and play;
- *notebook* folder: it contains step-by-step explanations of what was implemented, one notebook is available for basic QLearning, QNetworks and DQN;
- *results* folder: it contains the results obtained along with the different models configurations and the corresponding trained models to be used to play the game;
- *src_code* folder: it contains the implementation of each model and it is divided into the subsections of incremental complexity, which lead us to implementation of DQN. 

After the train is possible to run tests specifying which model to employ and save the result as gif. The code contain also a script to automatically run the train on a slurm server.

**Training**:
The training process was constrained by limited access to the university's computing resources and time. To experiment with various proxies, we opted for a simplified Deep Q-Network (DQN) using a Multilayer Perceptron (MLP) rather than a Convolutional Neural Network (CNN). We attempted to expedite training by leveraging vectorization capabilities of the GYM library, but this did not yield noticeable improvements.

A critical adjustment that significantly impacted performance was the strategy for handling instances where the snake in the game ate itself. We observed that ending the game or imposing a high penalty for this error was not effective. Instead, we initially programmed the snake to avoid eating itself during the early training phase, allowing the agent to explore more effectively. Once the agent learned to reach the food, we reintroduced the self-eating possibility. Eventually, the agent successfully learned to play the game without eating itself.

### Challenges and Solutions
- Setting the reward function.
Building our models incrementally has allowed us to have a control on what the
snake was doing and an execution time small enough to check results and apply
suitable corrections.
- Deciding the best way to create the input of the CNN.
We experimented with a multi-channel input approach to distinguish instances where the snake overlapped itself. In this method, each channel was dedicated to representing different elements: one for empty spaces, another for non-overlapping snake segments, and a third for overlapping segments of the snake. But this strategy did not yield noticeable improvements. Eventually we decide to use a simple 2d tensor as input where in each box store the count of overlapping segments 
- Choosing the strategy for epsilon decay.
The way in which we decay epsilon has a great impact on the results, but in particular
on the training time, even in this case having a simpler model, to use as comparison,
has been crucial.


### Code Repository / Demo Link
- GitHub Repository: [Github Link](https://github.com/erikalena/Deep-QNetworks)
- Presentation: [Link](https://github.com/erikalena/Deep-QNetworks/blob/Master/presentation_DQN.pdf)