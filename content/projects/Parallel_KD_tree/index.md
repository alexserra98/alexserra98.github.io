---
title: "Parallel KD_Tree"
date: 2023-09-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/projects/Parallel_KD_Tree/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
## Project: Parallel KD_Tree
### Overview
Parallel implementation of k-d tree using MPI, OpenMP and C++17
- **Key Technologies**: C++, MPI, OpenMP, Bash
{{<figure  src="images/kdtree.png" caption="Source: DALLE"  width="400" height="400">}}<br>
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

