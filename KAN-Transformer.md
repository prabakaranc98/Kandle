# Challenges of Integrating KANs into Transformers

## C1. Base Function Inefficiency

* KANs use **B-spline basis functions**.
* B-splines are not optimized for parallel execution on modern GPUs/TPUs.
* Results in **slower training and inference**.

## C2. High Parameter & Computational Cost

* KAN requires a **separate learnable function for every input-output connection**.
* The number of parameters and computations grows rapidly with model size.
* Leads to **high memory and computational overhead**.

## C3. Difficult Weight Initialization

* KANs employ **learnable activation functions** instead of fixed activations.
* Proper initialization becomes challenging.
* Poor initialization can cause **slow convergence and unstable training**.

# Solutions Proposed in KAN-Transformer

## S1. Rational Basis

* Replaces the traditional **B-spline basis functions** used in KANs with **rational functions**.
* Rational functions are composed of simple arithmetic operations (addition, multiplication, division), making them more GPU-friendly.
* Eliminates costly spline interval lookups and branching operations.
* Enables efficient CUDA implementation and faster training/inference.

## S2. Group KAN

* Instead of learning a separate activation function for every neuron connection, activation weights are **shared within groups of neurons**.
* Significantly reduces the number of learnable parameters and computations.
* Maintains model performance while improving efficiency.

## S3. Variance-Preserving Initialization

* Carefully initializes activation weights to maintain a stable activation variance across layers.
* Prevents vanishing and exploding activations during deep network training.
* Improves convergence and training stability.

