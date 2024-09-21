# Transformer OPUS

This project implements a Transformer model from scratch using PyTorch for text translation. The model is trained on the OPUS Books dataset, which contains multilingual text data. Key Transformer components such as multi-head attention, positional encoding, and layer normalization are implemented in this project.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Components](#components)
- [Results](#results)

## Overview

The **Transformer OPUS** project showcases the implementation of a transformer model, based on the architecture proposed in the paper *"Attention Is All You Need."* This model is designed for text translation tasks, and it was trained on the OPUS Books dataset, a multilingual corpus. The project focuses on core Transformer principles like self-attention, multi-head attention, and positional encodings.

## Features

- **Built from scratch**: All key components of the Transformer model are implemented using PyTorch.
- **Multi-head Attention**: For capturing different aspects of relationships between words.
- **Positional Encoding**: Injects information about the position of words in a sequence.
- **Layer Normalization**: Stabilizes and accelerates the training process.
- **Trained on OPUS Books dataset**: A multilingual dataset used for translation tasks.
- **Training Metrics**: Achieved a perplexity (PPL) of 23.0967 after three epochs.

## Model Architecture

- **Encoder-Decoder Architecture**: The model follows the standard transformer structure with an encoder and decoder.
- **Encoder**:
  - Multiple layers of self-attention followed by feedforward layers.
  - Positional encoding added to input embeddings.
- **Decoder**:
  - Layers of masked self-attention, multi-head attention over the encoder outputs, followed by feedforward layers.
- **Final Layer**: Softmax applied to the decoder output to generate translation tokens.

## Components

- **Multi-head Attention**: Computes self-attention across multiple heads to allow the model to focus on different word relationships in parallel.
  
  $$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V $$

- **Positional Encoding**: Since the Transformer model has no recurrence, positional encoding is used to inject sequence information. The encoding is added to the input embeddings.

  $$ PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right) $$
  $$ PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right) $$

- **Layer Normalization**: Ensures stable and efficient training by normalizing the inputs to each layer.

## Results

After training the model for three epochs on the OPUS Books dataset, the model achieved a perplexity (PPL) of **23.0967**. This indicates the model's ability to predict translations with reasonable accuracy given the small number of training epochs.
