# Machine Learning Portfolio

## Project: Transformer for Tabular Data
### Overview
The focus of this investigation was to adapt transformers, known for their success in natural language processing, to handle the unique challenges presented by tabular data. The core complication addressed was the constraint of each field in a table to a specific subset of vocabulary, necessitating an exploration of customizations in the transformer's architecture.
- **key Technologies** python, pythorch

### Project Details

The project involved a series of experiments with transformers applied to tabular data. Each field in the dataset was limited to a predefined set of tokens, a constraint atypical for standard transformer models. The primary approach experimented with was the modification of the transformer architecture to comply with these restrictions. This involved hardcoding the model to output tokens from the appropriate set for each table field and subsequently allowing the model to autonomously learn the association between fields and their respective token sets.

### Challenges and Solutions

1. **Hardcoding Vocabulary Constraints:**
   - *Challenge:* Initially, the model was hardcoded to select tokens from specific sets based on the field. This approach, however, did not yield the expected performance improvements.
   - *Solution:* A shift in strategy was implemented, allowing the transformer to learn the relationship between table fields and their corresponding token sets. This approach facilitated a deeper comprehension of each field's distribution and improved model performance.

2. **Training Dynamics and Local Minima:**
   - *Challenge:* The model tended to exploit easy prediction patterns, leading to entrapment in misleading local minima.
   - *Solution:* By restricting the model from using simple heuristics for predictions, it was encouraged to develop a more robust understanding of the data. This was achieved by shuffling the order of fields in the training datasets, preventing the model from relying on field position.

3. **Shuffled Fields for Improved Generalization:**
   - *Challenge:* The transformer model initially relied heavily on the position of each field for predictions.
   - *Solution:* Training datasets were altered to shuffle the order of fields, compelling the model to focus on the content of the fields rather than their position. This approach led to a significant improvement in the model's performance and generalization capabilities.

### Conclusion

The findings from this project offer valuable insights into the adaptability and potential of transformers in processing tabular data. The success of the model in learning field-token associations and overcoming the constraints of field-specific vocabularies marks a significant advancement in applying deep learning techniques to structured data. The results underscore the importance of training dynamics and data presentation in enhancing model performance and generalization.
### Code Repository / Demo Link
The repository is private because the code is part of the company's proprietary library

## Project: Parallel KD_Tree
### Overview
Parallel implementation of k-d tree using MPI, OpenMP and C++17
- **Key Technologies**: C++, MPI, OpenMP, Bash

### Project Details
A K-dimensional tree is a data structure widely used for partitioning and organizing
points in a k-dimensional space, they’re involved in many different applications
such as searches involving a multidimensional search key (e.g. range
searches and nearest neighbor searches). The task of the assignement was to write a program which takes as input a 2d dataset and compute over it the associated kd-tree.
The algorithm conist essentialy in these three steps
- picking round-robin an axis
- select the median point over that axis
- reiterate on the two halves at the left and right of the selected point
The algorithm stops when the region of the spaces that the program is parsing contains only one point.  
The output consist in a graphical representations of the tree and time elapsed during the computation.
The focus was to study and compare the scalability and the performance of this two parallelization framework, all the details and the commentary of the benchmarks are in the /report section
**Parallelization**:
MPI: The parallelization of the serial algorithm through the MPI interface consists
in distributing the recursive calls among the processes provided. The communication among processess is organized in the following way: 
starting from the master process - rank 0 in our implementation - each worker
keep half of the data set received and the send the other half to his son, once
all the processors has received their input data they proceed the computation
serially. At the end of this section each processor, with exception of the master,
starting from the bottom of the tree, start sending to his parent the tree computed
on the received portion of the data set. The parent, on his behalf, will
merge the received buffer with his tree so that eventually the master process
will have the complete tree.
OMP : The OpenMP implementation is similar to the MPI one although it follows a
much simpler scheme. The first call of the function is made by master thread
and then each recursive call is assigned to a new task.
### Challenges and Solutions
The primary challenge addressed here involves the lack of shared memory among processes in the MPI interface. To tackle this, a vector-based data structure was used for the implementation of a tree, where each node identifies its children by their index positions in the vector. This approach eliminates the dependence on pointers and specific memory addresses.

Another critical issue was managing the size of the buffer during inter-process communication. To avoid doubling the communication (which would negatively impact performance due to overhead), a preparatory phase was introduced. In this phase, a binary tree is computed corresponding to the communication scheme, where each node contains the size of the message to be received by the associated processor in the communication tree.

A third issue involves updating the indexes of the left and right children of tree nodes. Since pointers are replaced with indexes that depend on the array size, and the received tree is merged next to the previously computed tree, all child indexes must be shifted by the previously mentioned offset. While this update could happen each time a tree is received, it would result in multiple updates for some nodes.

To avoid redundant operations, an updating routine was implemented. During the dimension tree's construction, a "strides" vector is created. Each i-th element in this vector represents the nodes computed by the i-th rank, added to the final tree. These strides are then used to update the indexes just once when the entire tree has been received by rank 0, thus streamlining the process and reducing unnecessary computations.

### Code Repository / Demo Link
- GitHub Repository: [Github Link](https://github.com/alexserra98/KD_Tree/tree/main)
- Presentation: [Link](https://github.com/alexserra98/KD_Tree/tree/main/report)



## Project: Deep Q-Network for ATARI Snake Game

### Overview

The aim of the project was to implement Deep QNetworks, following the description contained in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and in particular to exploit it to make an agent learn how to play Snake game.

Instead of directly implementing DQN using CNNs, we progressively build up different models, starting from basic Qlearning algorithms and moving towards MLP, in order to deal with increasing complexity of the game. 
The code for all the trials we made is available, along with one simple version of Snake game, which can be played using all the models trained, from standard tabular methods to DQN.

- **Key Technologies**: Python, Pytorch, OpenAI Gym
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


## Project: Evolutionary architecture selection

### Overview
- This repository contains an implementation (in PyTorch) of the DENSER ( Deep Evolutionary Network Structured Representation) approach for automatic selection of a Convoluational Neural Network (CNN) architecture. 
- **Key Technologies**: Python, Pytorch, Genetic Algorithms

### Project Details 
The method proposed in the paper has to do with Neuroevolution, a field which deals with the automatic optimisation of ANN architectures and parametrization. The DENSER approach, in particular, combines Genetic Algorithms (GA) with Dynamic structured grammatical evolution (SGDE), which represents the genotype of each individual (a DNN in our case) through the use of a grammar, expressed in backus-naur form (BNF). (DSGE is slightly different from GE and SGE and these differences are explained in detail in the paper):
-   the GA level: encodes the macro-structure of the DNN (the "genes")
- the DSGE level: encodes the specific representation of each layer, along with its
parameters.

The grammar is used to dynamically generate a population of CNN architectures. Each individual of the population is then evaluated using a fitness function. As fitness function the accuracy of the CNN on a validation set is used. Evolution is performed in the following way:

* an initial population (of size 
) of randomly generated CNNs is created (following the rules of the given grammar)
then, for the number of generations we initially set, we repeat the following steps:

a fitness score is computed for each individual
* a pair of parents is chosen randomly from the fittest individuals
* through crossover operation we create two new individuals and we choose the longest one
* mutations are applied to the newly generated individual

The last three steps are repeated $n$ times for each generation, in order to obtain again a population of 
individuals.
The whole process is repeated until a CNN is found that meets the desired accuracy on the validation set.



### Challenges and Solutions
- How handle the input-output channels with a dynamic structure?
We record only output channels (random initialised) and we compute input channels
only when transform the encoding to a PyTorch net.
- How to avoid that the input shape for a certain layer is less then kernel size or is equal
to zero?
When we decode the network we drop all the layers that produce not valid shape.


### Outcomes
- Successfully replicated the DENSER algorithm, achieving comparable performance on the CIFAR-10 dataset with an average test accuracy of 94.13%.
- Demonstrated the algorithm's generalization capability by applying the evolved networks to MNIST and Fashion-MNIST datasets, achieving competitive accuracies.
- The replication provided insights into the strengths and limitations of evolutionary approaches in ANN design.

### Links
- GitHub Repository: [GitHub Link](https://github.com/Francesc0rtu/Genetics_ANN/tree/denser)
- Project Report: [Link to detailed project report](https://github.com/Francesc0rtu/Genetics_ANN/blob/denser/Deep_learning_slide.pdf)

## Project: Twitter (X at the time of writing) food popularity
### Overview
This report outlines the development and implementation of a machine learning model aimed at predicting the popularity of food-related tweets. The project was a part of the final exam for the Introduction to Machine Learning course 2021/22 at Units. It involved collecting a tweet corpus using the Twitter API and constructing a dataset using both raw features (like hashtags, day/hour of publishing) and content-based information derived through standard NLP preprocessing routines.
- **Key Technologies**: python, pytorch

### Project Details

- **Data Collection and Preparation:** The corpus of tweets was retrieved using the Twitter API. The dataset was composed of raw features and content-based information.
- **Popularity Metric:** The popularity of tweets was measured using a formula that combined the number of likes and retweets, normalized by the number of followers of the tweet's author. This metric was chosen to reflect the intrinsic ability of a tweet to generate interest, independent of the author's follower count.
- **Model Selection:** For the prediction task, the team selected Random Forest and Gradient Boosting models. The probability scores assigned to each tweet were binned, framing the task within a classification context.
- **Evaluation Metrics:** Standard metrics such as accuracy, F1 score, and precision were employed to assess the models' performance.

### Challenges and Solutions

- **Complexity of Popularity Phenomena:** 
  - *Challenge:* The test results were only moderately successful, attributed to the complex nature of popularity in social dynamics.
  - *Solution:* The report suggests the need for a deeper understanding of social dynamics to improve model performance. It also proposes the implementation of more refined tools capable of capturing these complex dynamics in future developments.
- **Intrinsic Randomness in Human Behavior:** 
  - *Challenge:* There is an acknowledgment that the randomness inherent in human nature might make it difficult to find a satisfying answer to the question posed by the project.
  - *Solution:* While no specific solution is outlined for this challenge, the report implies the need for future models to account for or adapt to this randomness.

### Code Repository / Demo Link
- GitHub Repository: [Github Link](https://github.com/alexserra98/Twitter-Food-Popularity)

---

