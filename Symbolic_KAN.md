# Literature Review

## Symbolic–KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning

**Paper:** https://arxiv.org/pdf/2603.23854

### Problem Statement

Kolmogorov-Arnold Networks (KANs) have demonstrated strong function approximation capabilities and improved training dynamics compared to traditional neural networks. However, the learned functional representations often become increasingly complex and opaque, making them difficult to interpret. This limits the usefulness of KANs in applications where symbolic understanding and human interpretability are important.

### Methodology

Symbolic-KAN preserves the Kolmogorov-Arnold principle that multivariate functions can be represented through compositions of univariate functions. Instead of assigning a separate learnable function to every edge as in standard KANs, Symbolic-KAN first learns scalar projections of the input features through trainable linear combinations. Symbolic primitives are then applied to these learned projections, resulting in a more compact, interpretable, and symbolic representation.

## Unit Construction in Symbolic-KAN

### Step 1: Learned Scalar Projections

Each unit creates multiple learned scalar projections of the input using linear transformations:

\[
s = w^T h + b
\]

where:

- \(h\) is the input activation vector
- \(w\) is a learnable weight vector
- \(b\) is a learnable bias term

These projections represent different combinations of upstream features.

### Step 2: Symbolic Primitive Transformation

Each projection is passed through a mixture of symbolic primitives from a predefined library, such as:

- \(x\)
- \(x^2\)
- \(x^3\)
- \(\sin(x)\)
- \(\cos(x)\)
- \(\tanh(x)\)
- \(e^x\)
- \(\log(1+|x|)\)

The transformed output of an edge is represented as a weighted combination of these primitives.

### Step 3: Primitive Selection

Primitive-selection weights (\(\alpha\)) determine which symbolic primitive best represents the relationship in the data.

Initially, multiple primitives contribute to the output. During training, the coefficients are encouraged to become one-hot, resulting in the selection of a single symbolic primitive.

### Step 4: Edge Selection

Multiple candidate projections are generated for each unit.

Edge-selection weights (\(\eta\)) determine which projection is most relevant. Similar to primitive selection, these weights are gradually driven toward a one-hot representation during training.

### Step 5: Symbolic Hardening

After training:

- One projection is selected.
- One symbolic primitive is selected.

This transforms the unit into a simple symbolic expression that is easy to interpret and export.

### Final Unit Output

The final symbolic representation of a unit is:

$$
h_k = A f(\gamma s + \beta) + B
$$

where:

- \(s\) = selected scalar projection
- \(f\) = selected symbolic primitive
- \(\gamma\) = input scaling parameter
- \(\beta\) = input shifting parameter
- \(A\) = output scaling parameter
- \(B\) = output shifting parameter

Thus, each unit ultimately represents a single symbolic operation applied to a learned projection of the input features, enabling interpretable symbolic reasoning while retaining the expressive power of neural networks.

### Key Idea

A Symbolic-KAN unit learns several candidate feature projections and symbolic functions, then gradually prunes them until only a single projection and a single symbolic primitive remain, producing a compact and interpretable symbolic expression.
