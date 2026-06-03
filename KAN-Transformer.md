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

