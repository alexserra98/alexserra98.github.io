---
title: "MCQA Benchmark"
date: 2023-09-18T11:30:03+00:00
weight: 1
mathjax: true
editPost:
    URL: "https://alexserra98.github.io/projects/MCQA-Benchmark/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Overview
The [MCQA_Benchmark](https://github.com/alexserra98/MCQA_Benchmark) serves as a comprehensive benchmark for Language Model (LLM) performance on multiple-choice Q&A datasets. It primarily utilizes various methods from [DADApy](https://dadapy.readthedocs.io/en/latest/index.html), enabling the analysis of model hidden states through geometric tools like intrinsic dimension calculation and advanced clustering techniques.

- **Key Technologies**: Python, PyTorch, DADApy, Hugging Face Transformers, Dataset Management, SQLite

## Project Details
The inference configuration allows users to choose a dataset and model from the Hugging Face Hub, including the number of shots for few-shot learning. The pipeline processes the instances, collects hidden states, and stores them using h5py. Each instance's output includes metadata, organized in a SQLite database with the hidden states tensor hash code as the primary key. In the metrics configuration, users can select hidden states datasets and metrics to calculate, such as intrinsic dimension, density peak clustering, and neighbor overlap (both labeled and regular). Additionally, the project includes a Slurm runner script, currently tested only on the developer's university cluster.


