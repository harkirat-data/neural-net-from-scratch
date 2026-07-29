# Neural Network From Scratch (NumPy)

A feedforward neural network built entirely from scratch in NumPy — no PyTorch, no TensorFlow, no autograd. The goal: understand exactly what happens mathematically during forward propagation and weight updates, at the level of individual array operations, instead of treating a framework's `.backward()` call as a black box.

**Architecture:** 2 input features → 2-neuron hidden layer → 1 output neuron.
**Problem type:** Regression — predicting a continuous value (LPA) from CGPA and profile score.

## Dataset

| CGPA | Profile Score | LPA |
|-----:|--------------:|----:|
| 8 | 8 | 4 |
| 7 | 9 | 5 |
| 6 | 10 | 6 |
| 5 | 12 | 7 |

A small, hand-built dataset used to trace every value through the network by hand and confirm the math checks out at each step.

## What's implemented

**Parameter initialization** — `initialize_parameters` builds a weights/bias dictionary for any given `layer_dims` list, so the same function works regardless of how many layers or units you specify.

**Forward propagation** — `linear_forward` computes `Z = WᵀA + b` for a single layer; `L_layer_forward` chains this across every layer in the network, passing the activation from one layer as the input to the next.

**Gradient derivation and weight updates** — the partial derivative of the loss with respect to every single weight and bias was derived by hand using the chain rule, specifically for this 2→2→1 architecture, and then implemented directly in `update_parameters`. Each line in that function corresponds to one hand-worked derivative — e.g. `∂L/∂W1[0][0]` traced back through the hidden layer to the output error. This is what backpropagation actually *is* at its core, done explicitly rather than through an automatic differentiation engine.

**Training loop** — iterates over epochs, running every row through forward propagation, computing squared error, updating parameters via the derived gradients, and tracking mean loss per epoch.

## Limitations

- No activation function is applied between layers — every layer is currently a linear transformation, so the network doesn't yet capture nonlinear relationships.
- The gradient derivations in `update_parameters` are specific to this 2→2→1 architecture, hardcoded per-parameter — not a generalized backward-pass function that scales to arbitrary layer sizes.
- No train/test split, batching, or learning rate scheduling yet.

## Sample output

```
Epoch 1 Loss: 25.29
Epoch 2 Loss: 18.16
Epoch 3 Loss: 9.21
Epoch 4 Loss: 3.07
Epoch 5 Loss: 1.29
```

## Stack

Python, NumPy, pandas

---

# Keras Implementation (`model-with-libraries.ipynb`)

The same regression problem, same dataset, same 2→2→1 architecture — this time built with Keras instead of hand-derived gradients, as a direct comparison against the from-scratch version above.

**Architecture:** `Sequential` model with two `Dense` layers — a 2-unit hidden layer and a 1-unit output layer, both using **linear activation**, matching the from-scratch model's lack of nonlinearity so the two are comparable.

**Weight initialization** — Keras's default weights were manually overridden (`model.set_weights`) with the same starting values (0.1) used in the from-scratch model, so both models begin training from an identical point.

**Training** — compiled with the Adam optimizer (learning rate 0.001) and mean squared error loss, trained for 75 epochs with a batch size of 1 (one row at a time, matching the from-scratch training loop).

**Result** — loss dropped from ~27.9 to ~1.48 over 75 epochs, consistent with the from-scratch model's loss trajectory over its 5 epochs, confirming the manually derived gradients in the from-scratch implementation behave the way an autograd-based framework's gradients do.
