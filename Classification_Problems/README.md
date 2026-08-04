# Neural Network From Scratch (NumPy) — Classification

A feedforward neural network built entirely from scratch in NumPy — no PyTorch, no TensorFlow, no autograd. Same approach as the regression project: derive the math by hand and implement every step explicitly.

**Architecture:** 2 input features → 2-neuron hidden layer → 1 output neuron, with **sigmoid activation applied at every layer** (a real difference from the regression version, which was fully linear).
**Problem type:** Binary classification — predicting placement (0/1) from CGPA and profile score.

## Dataset

| CGPA | Profile Score | Placed |
|-----:|--------------:|-------:|
| 8 | 8 | 1 |
| 7 | 9 | 1 |
| 6 | 10 | 0 |
| 5 | 5 | 0 |

Same small, hand-traceable format as the regression dataset, with a binary label instead of a continuous one.

## What's implemented

**Parameter initialization** — same generalized `initialize_parameters` function used in the regression project, works for any `layer_dims`.

**Forward propagation with sigmoid** — `linear_forward` now applies `sigmoid(Z)` after the linear step at every layer, so the network's output is bounded between 0 and 1, as required for binary classification. `L_layer_forward` chains this across layers.

**Loss function** — binary cross-entropy (`-y·log(ŷ) - (1-y)·log(1-ŷ)`), printed per sample and averaged per epoch, replacing the mean squared error used in the regression version.

**Gradient derivation and weight updates** — partial derivatives were hand-derived for this architecture and hardcoded in `update_parameters`, the same approach as the regression project.

**Training loop** — same structure as the regression version: 50 epochs, one row at a time, tracking mean loss per epoch.

## Limitations

- The gradient formulas in `update_parameters` appear to carry over the same chain-rule structure used for the (fully linear) regression model, without an explicit term for the hidden layer's sigmoid derivative — worth double-checking against the correct binary cross-entropy + sigmoid gradient before relying on this for anything beyond the toy dataset.
- Training loss barely moves across all 50 epochs (stays close to 0.693, which is `ln(2)` — the loss a model gets by predicting 0.5 for everything). On this dataset the model isn't learning to separate the classes yet.
- Same architecture-specific limitation as the regression project: the update rules are hardcoded for this exact 2→2→1 shape, not a generalized backward pass.
- No train/test split, batching, or learning rate scheduling.

## Sample output

```
Epoch 1  Loss - 0.6941
Epoch 25 Loss - 0.6942
Epoch 50 Loss - 0.6943
```

## Stack

Python, NumPy, pandas

---

# Keras Implementation (`model-with-libraries.ipynb`)

Same classification problem, same dataset, same 2→2→1 architecture with sigmoid activations — built with Keras for direct comparison against the from-scratch version above.

**Architecture:** `Sequential` model with two `Dense` layers, both using sigmoid activation, matching the from-scratch model's structure exactly.

**Weight initialization** — Keras's default weights were manually overridden (`model.set_weights`) with the same starting values (0.1) used in the from-scratch model, so both models begin training from an identical point.

**Training** — compiled with the Adam optimizer (learning rate 0.001) and binary cross-entropy loss, trained for 50 epochs with a batch size of 1, matching the from-scratch training loop exactly.

**Result** — loss also stays close to 0.693 across all 50 epochs, consistent with the from-scratch model. Both implementations show the same plateau, which suggests the flat loss is a property of this dataset/setup (likely too few samples and epochs for the model to move away from the random baseline) rather than an error unique to one implementation.
