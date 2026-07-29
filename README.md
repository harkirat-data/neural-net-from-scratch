# Neural Networks: From Scratch vs Framework Implementations

Each problem type is implemented two ways — once manually in NumPy (forward propagation and gradient updates derived and coded by hand, no autograd), and once using a high-level framework (Keras) for comparison. The goal is to understand the underlying math first, then see how the same problem is expressed once a framework handles the mechanics.

## Structure

### `Regression_Problems/`
Predicting a continuous value (LPA from CGPA and profile score).
- `model-from-scratch.ipynb` — manual NumPy implementation
- `model-with-libraries.ipynb` — Keras implementation
- See [Regression_Problems/README.md](./Regression_Problems/README.md) for details on the from-scratch implementation.

### `Classification_Problems/`
In progress — same from-scratch vs framework structure, applied to a classification task.

## Stack

Python, NumPy, pandas, Keras/TensorFlow
