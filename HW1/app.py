# My attempt to not use AI, and write down the logical steps
# I've used pytorch, and have some experience with these libraries
# and a jist from NLP and AI

import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
    api_name="predict"
)

demo.launch()


# transform images to 32x32 grayscale images

# Dataset loader returns (x,y)
# x is a 1024-d float vector
# y is a number 0 or 1


# Split the dataset into training and testing sets
# Training set: 80% of the data
# Testing set: 20% of the data

# Train two models:
# 1. Logistic regression model
# 2. Neural network model

# Logistic regression model:
# Single layer neural network with 1024 input nodes and 1 output node

# Neural network model:
# Implement a configurable MLP with:
# Depth: 1, 2, or 3 hidden layers
# Width: number of units per layer 32–256
# Activation: ReLU (default)
# output layer: 2 logits


# Train each model on the training set
# Train both models the same way


# Show through confiugurable slider how depth matters

# Wrap with Gradio and use Hugging Face 