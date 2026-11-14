# Transformer Implementation and Architecture Analysis

1. Overview

This project implements and evaluates Transformer-based architectures using PyTorch. It explores scaled dot-product attention, Transformer encoder blocks, Human Activity Recognition (HAR) classification, and character-level language modelling using the Tiny Shakespeare dataset.

The objective is to understand the Transformer architecture through implementation from scratch, unit testing, model training, and experimental analysis.


Task A: Implemented scaled dot-product attention from scratch with optional masking.

            Query, Key and Value projections.
            scaled attention score calculation.
            Softmax-based attention weights.
            Optional attention masking.
            Causal masking to prevent access to future token


Task B: Implemented a Transformer encoder using:

          Multi-head self-attention.
          Pre-Layer Normalization.
          Feed-forward neural network.
          Residual connections.
          Positional encodings.

Task C: The Transformer encoder was applied to the UCI Human Activity Recognition (HAR) dataset.

          Dataset: UCI HAR.
          Input features: 561 engineered features.
          Patch size: 8.
          Number of tokens: 70.
          Embedding dimension: 128.
          Transformer layers: 4.
          Positional encoding: No positional encoding, sinusoidal encoding and learned encoding.

Task D: character-level language model was implemented using the Tiny Shakespeare dataset.

          Character-level tokenisation.
          Token embeddings.
          Positional embeddings.
          Causal self-attention.
          Transformer decoder blocks.
          Autoregressive text generation.

Task E: Experimental Analysis and Ablation

          Comparison of different numbers of attention heads.
          Comparison of Transformer performance with and without causal masking.
          Analysis of attention heatmaps.
          Training time and GPU memory analysis across different context lengths.
          Key-Value (KV) cache performance evaluation.
          Comparison with an LSTM character-level language mode


# Execution Environment

The experiments were conducted using:

       Component: Configuration
       Platform: Google Colab
       Programming Language: Python
       Deep Learning Framework: PyTorch
       Hardware Accelerator: NVIDIA Tesla T4 GPU
       Notebook Environment: Google Colab Jupyter Notebook


Run time:  

Problem 1: 5 Minutes 

Problem 2: 15 Minutes
