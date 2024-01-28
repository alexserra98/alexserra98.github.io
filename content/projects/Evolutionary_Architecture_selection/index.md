---
title: "Evolutionary architecture selection"
date: 2023-09-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/projects/Evolutionary_architecture_selection/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
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