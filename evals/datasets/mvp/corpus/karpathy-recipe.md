---
title: Karpathy — A Recipe for Training Neural Networks
source: https://karpathy.github.io/2019/04/25/recipe/
date: 2019-04-25
note: Condensed summary of Karpathy's blog post; used as retrieval-quality corpus material.
---

# A Recipe for Training Neural Networks

## Introduction

A gap exists between understanding neural network theory and achieving competitive results. Following a deliberate process helps avoid common mistakes and speeds debugging.

## Two Key Observations

### 1) Neural net training is a leaky abstraction

Libraries suggest neural networks are plug-and-play, but they require deep understanding. Unlike libraries like Requests that abstract away complexity cleanly, neural nets fail silently when misconfigured. "Batch norm does not magically make it converge faster. RNNs don't magically let you 'plug in' text."

### 2) Neural net training fails silently

Syntax errors get caught easily, but logical errors in neural network training persist quietly. A misconfigured model trains but performs suboptimally. This is why a "fast and furious" approach does not work, and patience plus attention to detail matter most.

## The Recipe

### 1. Become one with the data

Before touching code, spend hours examining thousands of examples. Look for:
- Duplicates and corrupted data
- Imbalances and biases
- Feature importance (local vs. global context)
- Label noise
- Spurious variation

Write filtering code to visualize distributions and outliers, which almost always uncovers bugs in data quality or preprocessing.

### 2. Set up the end-to-end training/evaluation skeleton + get dumb baselines

Build a simple training pipeline with a basic model (linear classifier or tiny ConvNet). Then validate correctness through systematic experiments.

Tips and tricks:
- **Fix a random seed** — ensures reproducibility.
- **Simplify** — disable data augmentation initially.
- **Add significant digits** — evaluate on full test sets, not batches.
- **Verify loss @ init** — check initial loss matches theoretical expectations.
- **Init well** — set final layer biases appropriately (e.g., bias = log(positive_ratio/negative_ratio) for imbalanced data).
- **Human baseline** — compare against your own accuracy.
- **Input-independent baseline** — verify the model learns from actual inputs.
- **Overfit one batch** — confirm the model can reach zero loss on a few examples.
- **Verify decreasing training loss** — increase capacity and confirm improvement.
- **Visualize just before the net** — inspect raw data entering the model.
- **Visualize prediction dynamics** — watch how predictions evolve during training.
- **Use backprop to chart dependencies** — verify gradients flow correctly.
- **Generalize a special case** — write specific implementations first, then generalize.

### 3. Overfit

Now that you trust your pipeline, find a model large enough to overfit the training set, then regularize.

Tips and tricks:
- **Picking the model**: don't be a hero. Copy proven architectures (e.g., ResNet-50) rather than inventing custom ones.
- **Adam is safe**: use learning rate of 3e-4 initially; Adam forgives poor hyperparameters better than SGD.
- **Complexify only one at a time**: add signals/features incrementally.
- **Do not trust learning rate decay defaults**: disable decay initially and tune at the end.

### 4. Regularize

With a model that overfits training data, reduce overfitting while maintaining validation performance.

Tips and tricks:
- **Get more data** — the best regularization is real additional training data.
- **Data augment** — use aggressive augmentation.
- **Creative augmentation** — employ domain randomization, simulation, or synthetic data.
- **Pretrain** — leverage pretrained networks when possible.
- **Stick with supervised learning** — unsupervised pretraining rarely helps in modern vision.
- **Smaller input dimensionality** — remove spurious features.
- **Smaller model size** — use domain knowledge to constrain architecture.
- **Decrease batch size** — smaller batches provide implicit regularization via batch normalization.
- **Drop** — add dropout (use spatial dropout for ConvNets).
- **Weight decay** — increase L2 penalty.
- **Early stopping** — halt training when validation loss plateaus.
- **Try a larger model** — counterintuitively, larger models with early stopping sometimes outperform smaller ones.

Visualize first-layer weights to ensure they show meaningful patterns like edges, not noise.

### 5. Tune

Explore model architectures and hyperparameters systematically.

Tips and tricks:
- **Random over grid search** — random search explores hyperparameter space more efficiently than grid search when parameters have different importance.
- **Hyper-parameter optimization** — Bayesian optimization tools exist, but manual exploration remains practical.

### 6. Squeeze out the juice

Final performance gains from remaining techniques:

- **Ensembles** — nearly guaranteed 2% accuracy improvement.
- **Leave it training** — networks often improve for surprisingly long periods; extended training can yield unexpected gains.

## Conclusion

Following this recipe provides understanding of the technology, dataset, and problem. With validated infrastructure and incremental complexity exploration, you're prepared for advanced experimentation and state-of-the-art results.
