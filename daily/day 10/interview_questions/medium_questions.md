# Day 10 - Medium Interview Questions

## 1. Derive the mathematical formulation of L2 regularization and explain how it affects the gradient descent update rule.

**Answer:**

**L2 Regularization Formulation:**

The total loss with L2 regularization is:

$$L_{total} = L_{data} + \lambda \sum_{i} w_i^2$$

Where:
- $L_{data}$ is the original data loss (e.g., cross-entropy, MSE)
- $\lambda$ is the regularization strength
- $\sum_{i} w_i^2$ is the sum of squared weights

**Gradient Computation:**

The gradient of the total loss with respect to weight $w_i$ is:

$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + \frac{\partial}{\partial w_i}\left(\lambda \sum_{j} w_j^2\right)$$

Since $\frac{\partial}{\partial w_i}(\lambda \sum_{j} w_j^2) = 2\lambda w_i$:

$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + 2\lambda w_i$$

**Gradient Descent Update Rule:**

Standard gradient descent update:
$$w_i^{new} = w_i^{old} - \alpha \frac{\partial L_{total}}{\partial w_i}$$

Substituting the gradient:
$$w_i^{new} = w_i^{old} - \alpha \left(\frac{\partial L_{data}}{\partial w_i} + 2\lambda w_i^{old}\right)$$

Rearranging:
$$w_i^{new} = w_i^{old} - \alpha \frac{\partial L_{data}}{\partial w_i} - 2\alpha\lambda w_i^{old}$$

$$w_i^{new} = w_i^{old}(1 - 2\alpha\lambda) - \alpha \frac{\partial L_{data}}{\partial w_i}$$

**Key Observations:**

1. **Weight Decay Term:** The factor $(1 - 2\alpha\lambda)$ causes weights to decay at each step, even without the data gradient.

2. **Multiplicative Decay:** Weights are multiplied by $(1 - 2\alpha\lambda) < 1$ at each step, causing exponential decay toward zero.

3. **Gradient Penalty:** The $2\lambda w_i$ term adds a penalty proportional to the weight magnitude, pushing large weights harder toward zero.

**Mathematical Example:**

For $\alpha = 0.01$ and $\lambda = 0.0001$:
- Decay factor: $1 - 2 \times 0.01 \times 0.0001 = 0.999998$
- At each step, weights are multiplied by 0.999998 (slight decay)
- Over many steps, this cumulative effect shrinks weights

---

## 2. Derive the mathematical formulation of L1 regularization and explain why it creates sparsity (sets weights to exactly zero).

**Answer:**

**L1 Regularization Formulation:**

The total loss with L1 regularization is:

$$L_{total} = L_{data} + \lambda \sum_{i} |w_i|$$

**Gradient Computation:**

The gradient of the total loss with respect to weight $w_i$ is:

$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + \frac{\partial}{\partial w_i}\left(\lambda \sum_{j} |w_j|\right)$$

The derivative of $|w_i|$ is:
$$\frac{\partial |w_i|}{\partial w_i} = \begin{cases} 
+1 & \text{if } w_i > 0 \\
-1 & \text{if } w_i < 0 \\
\text{undefined} & \text{if } w_i = 0
\end{cases} = \text{sign}(w_i)$$

Therefore:
$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

**Gradient Descent Update Rule:**

$$w_i^{new} = w_i^{old} - \alpha \left(\frac{\partial L_{data}}{\partial w_i} + \lambda \cdot \text{sign}(w_i^{old})\right)$$

$$w_i^{new} = w_i^{old} - \alpha \frac{\partial L_{data}}{\partial w_i} - \alpha\lambda \cdot \text{sign}(w_i^{old})$$

**Why L1 Creates Sparsity:**

**Key Insight:** The L1 penalty gradient is **constant** ($\lambda$ or $-\lambda$), regardless of weight magnitude, unlike L2 which is proportional to weight.

**Mathematical Proof of Sparsity:**

Consider a weight $w_i$ that is small. The update rule becomes:

$$w_i^{new} = w_i^{old} - \alpha \frac{\partial L_{data}}{\partial w_i} - \alpha\lambda \cdot \text{sign}(w_i^{old})$$

If $|\frac{\partial L_{data}}{\partial w_i}| < \lambda$ (data gradient is small), then:

- If $w_i > 0$: $w_i^{new} = w_i^{old} - \text{small} - \alpha\lambda < w_i^{old}$
- If $w_i < 0$: $w_i^{new} = w_i^{old} - \text{small} + \alpha\lambda > w_i^{old}$

The constant penalty $\alpha\lambda$ will push $w_i$ toward zero. Once $w_i$ crosses zero, the sign changes, and the penalty pushes it back. This creates a "tug-of-war" that can push $w_i$ to exactly zero.

**Subgradient at Zero:**

At $w_i = 0$, the subgradient of $|w_i|$ is $[-1, 1]$. If:
$$\left|\frac{\partial L_{data}}{\partial w_i}\right| < \lambda$$

Then the optimal subgradient can be chosen such that:
$$\frac{\partial L_{total}}{\partial w_i} = \frac{\partial L_{data}}{\partial w_i} + \lambda \cdot g = 0$$

For some $g \in [-1, 1]$. This means $w_i = 0$ is optimal!

**Comparison with L2:**

- **L2:** Gradient is $2\lambda w_i$ (proportional to weight). As $w_i \to 0$, gradient $\to 0$, so weight never reaches exactly zero.
- **L1:** Gradient is $\lambda \cdot \text{sign}(w_i)$ (constant). Even as $w_i \to 0$, gradient remains $\lambda$, allowing weight to cross zero.

**Mathematical Example:**

For a weight $w_i = 0.01$ with small data gradient $\frac{\partial L_{data}}{\partial w_i} = 0.001$ and $\lambda = 0.01$:

**L2 Update:**
$$w_i^{new} = 0.01 - 0.01 \times (0.001 + 2 \times 0.01 \times 0.01) = 0.01 - 0.00012 = 0.00988$$
(Shrinks but doesn't reach zero)

**L1 Update:**
$$w_i^{new} = 0.01 - 0.01 \times (0.001 + 0.01 \times 1) = 0.01 - 0.00011 = 0.00989$$
(Similar, but if data gradient is smaller, can cross zero)

---

## 3. Explain the mathematical relationship between regularization strength (lambda) and the bias-variance tradeoff. Derive the optimal lambda for a simple case.

**Answer:**

**Bias-Variance Decomposition:**

The expected prediction error can be decomposed as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \text{Irreducible Error}$$

Where:
- **Bias:** $\text{Bias}(\hat{f}(x)) = \mathbb{E}[\hat{f}(x)] - f(x)$ (systematic error)
- **Variance:** $\text{Var}(\hat{f}(x)) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$ (sensitivity to data)

**Effect of Regularization:**

Regularization with strength $\lambda$ affects both bias and variance:

**As $\lambda$ increases:**
- **Bias increases:** Model becomes simpler, may underfit
- **Variance decreases:** Model is less sensitive to training data

**As $\lambda$ decreases:**
- **Bias decreases:** Model can fit training data better
- **Variance increases:** Model becomes more sensitive (overfitting)

**Mathematical Formulation:**

For ridge regression (L2 regularization), the solution is:

$$\hat{\beta}_\lambda = (X^T X + \lambda I)^{-1} X^T y$$

**Bias:**
$$\text{Bias}(\hat{\beta}_\lambda) = \mathbb{E}[\hat{\beta}_\lambda] - \beta = -(X^T X + \lambda I)^{-1} \lambda \beta$$

As $\lambda \to 0$: Bias $\to 0$ (unbiased)
As $\lambda \to \infty$: Bias $\to -\beta$ (highly biased toward zero)

**Variance:**
$$\text{Var}(\hat{\beta}_\lambda) = \sigma^2 (X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1}$$

As $\lambda \to 0$: Variance $\to \sigma^2 (X^T X)^{-1}$ (can be large if $X^T X$ is ill-conditioned)
As $\lambda \to \infty$: Variance $\to 0$ (weights shrink to zero)

**Optimal Lambda (Simple Case):**

For ridge regression with known noise variance $\sigma^2$, the optimal $\lambda$ minimizes:

$$MSE(\lambda) = \text{Bias}^2(\lambda) + \text{Var}(\lambda)$$

**Derivation (Simplified):**

Assuming $X^T X = I$ (orthonormal features) and true parameters $\beta$:

**Bias:**
$$\text{Bias}^2 = \lambda^2 \beta^T (I + \lambda I)^{-2} \beta = \frac{\lambda^2}{(1+\lambda)^2} \|\beta\|^2$$

**Variance:**
$$\text{Var} = \sigma^2 \text{tr}((I + \lambda I)^{-2}) = \frac{\sigma^2 p}{(1+\lambda)^2}$$

Where $p$ is the number of parameters.

**Total MSE:**
$$MSE(\lambda) = \frac{\lambda^2 \|\beta\|^2 + \sigma^2 p}{(1+\lambda)^2}$$

**Optimal Lambda:**

Taking derivative and setting to zero:

$$\frac{d}{d\lambda} MSE(\lambda) = 0$$

Solving (simplified):
$$\lambda^* \approx \frac{\sigma^2 p}{\|\beta\|^2}$$

**Key Insights:**

1. **Optimal lambda depends on:**
   - Noise level ($\sigma^2$): More noise → more regularization
   - Model complexity ($p$): More parameters → more regularization
   - Signal strength ($\|\beta\|^2$): Stronger signal → less regularization needed

2. **Bias-Variance Tradeoff:**
   - Small $\lambda$: Low bias, high variance (overfitting)
   - Large $\lambda$: High bias, low variance (underfitting)
   - Optimal $\lambda$: Balances both

**In Practice:**

The optimal $\lambda$ is found via:
- Cross-validation
- Grid search
- Monitoring validation loss

---

## 4. Derive the gradient formulas for backpropagation with L2 regularization. Show how regularization affects gradient flow through the network.

**Answer:**

**Setup:**

Consider a neural network with L2 regularization. The total loss is:

$$L_{total} = L_{data} + \frac{\lambda}{2} \sum_{l=1}^{L} \sum_{i,j} (W_{ij}^{[l]})^2$$

Where the factor $\frac{1}{2}$ is included for mathematical convenience (cancels the 2 in the derivative).

**Forward Pass (Unchanged):**

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = \sigma(Z^{[l]})$$

**Backward Pass with Regularization:**

**Step 1: Gradient w.r.t. Output Layer Weights**

For the output layer $L$:

$$\frac{\partial L_{total}}{\partial W_{ij}^{[L]}} = \frac{\partial L_{data}}{\partial W_{ij}^{[L]}} + \frac{\partial}{\partial W_{ij}^{[L]}}\left(\frac{\lambda}{2} \sum_{k,m} (W_{km}^{[L]})^2\right)$$

The regularization term:
$$\frac{\partial}{\partial W_{ij}^{[L]}}\left(\frac{\lambda}{2} \sum_{k,m} (W_{km}^{[L]})^2\right) = \lambda W_{ij}^{[L]}$$

Therefore:
$$\frac{\partial L_{total}}{\partial W_{ij}^{[L]}} = \frac{\partial L_{data}}{\partial W_{ij}^{[L]}} + \lambda W_{ij}^{[L]}$$

**In Matrix Form:**

$$\frac{\partial L_{total}}{\partial W^{[L]}} = \frac{\partial L_{data}}{\partial W^{[L]}} + \lambda W^{[L]}$$

**Step 2: Gradient w.r.t. Hidden Layer Weights**

For any hidden layer $l$:

$$\frac{\partial L_{total}}{\partial W^{[l]}} = \frac{\partial L_{data}}{\partial W^{[l]}} + \lambda W^{[l]}$$

**Step 3: Complete Gradient Flow**

The data loss gradient flows through the network via chain rule:

$$\frac{\partial L_{data}}{\partial W^{[l]}} = \frac{\partial L_{data}}{\partial A^{[L]}} \cdot \prod_{k=l+1}^{L} \frac{\partial A^{[k]}}{\partial Z^{[k]}} \cdot \frac{\partial Z^{[k]}}{\partial A^{[k-1]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$$

**With Regularization Added:**

$$\frac{\partial L_{total}}{\partial W^{[l]}} = \left(\frac{\partial L_{data}}{\partial A^{[L]}} \cdot \prod_{k=l+1}^{L} \frac{\partial A^{[k]}}{\partial Z^{[k]}} \cdot \frac{\partial Z^{[k]}}{\partial A^{[k-1]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}\right) + \lambda W^{[l]}$$

**Effect on Gradient Flow:**

**1. Gradient Magnitude:**
- Regularization adds $\lambda W^{[l]}$ to every weight gradient
- Large weights get larger penalty gradients
- This pushes weights toward zero

**2. Weight Updates:**
$$W^{[l]} \leftarrow W^{[l]} - \alpha \left(\frac{\partial L_{data}}{\partial W^{[l]}} + \lambda W^{[l]}\right)$$

$$W^{[l]} \leftarrow W^{[l]}(1 - \alpha\lambda) - \alpha \frac{\partial L_{data}}{\partial W^{[l]}}$$

**3. Cumulative Effect:**

Over $T$ training steps, if data gradient is zero:
$$W^{[l]}(T) = W^{[l]}(0) \cdot (1 - \alpha\lambda)^T$$

This exponential decay ensures weights shrink over time.

**Mathematical Example:**

For a 3-layer network with ReLU:

**Layer 3 (Output):**
$$\frac{\partial L_{total}}{\partial W^{[3]}} = dZ^{[3]} \cdot (A^{[2]})^T + \lambda W^{[3]}$$

**Layer 2:**
$$\frac{\partial L_{total}}{\partial W^{[2]}} = dZ^{[2]} \cdot (A^{[1]})^T + \lambda W^{[2]}$$

Where $dZ^{[2]} = (W^{[3]})^T dZ^{[3]} \odot (Z^{[2]} > 0)$

**Layer 1:**
$$\frac{\partial L_{total}}{\partial W^{[1]}} = dZ^{[1]} \cdot X^T + \lambda W^{[1]}$$

Where $dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot (Z^{[1]} > 0)$

**Key Insight:**

Regularization affects every layer independently by adding $\lambda W^{[l]}$ to the gradient. This:
- Doesn't affect gradient flow through activations
- Adds a constant "shrinkage" term at each layer
- Ensures all weights decay toward zero over time

---

## 5. Explain the mathematical relationship between L1 and L2 regularization from an optimization perspective. Compare their effect on the loss landscape.

**Answer:**

**Optimization Perspective:**

Both L1 and L2 regularization modify the loss function to create a constrained optimization problem:

**Unconstrained Problem:**
$$\min_{w} L_{data}(w)$$

**L2 Regularized Problem:**
$$\min_{w} L_{data}(w) + \lambda \|w\|_2^2$$

This is equivalent to:
$$\min_{w} L_{data}(w) \quad \text{subject to } \|w\|_2^2 \leq t$$

Where $t$ is determined by $\lambda$ (Lagrange multiplier).

**L1 Regularized Problem:**
$$\min_{w} L_{data}(w) + \lambda \|w\|_1$$

This is equivalent to:
$$\min_{w} L_{data}(w) \quad \text{subject to } \|w\|_1 \leq t$$

**Geometric Interpretation:**

**L2 Constraint (Ridge):**
- Constraint: $\|w\|_2^2 = \sum w_i^2 \leq t$ (circle/sphere in 2D/3D)
- **Smooth constraint:** Differentiable everywhere
- **Isotropic:** Treats all directions equally
- **Solution:** Usually interior point (weights rarely exactly zero)

**L1 Constraint (Lasso):**
- Constraint: $\|w\|_1 = \sum |w_i| \leq t$ (diamond/octahedron in 2D/3D)
- **Non-smooth constraint:** Has corners/edges
- **Anisotropic:** Favors axis-aligned solutions
- **Solution:** Often on corners (weights exactly zero)

**Loss Landscape Analysis:**

**L2 Regularization:**
$$L_{total} = L_{data}(w) + \lambda \sum w_i^2$$

**Gradient:**
$$\nabla L_{total} = \nabla L_{data} + 2\lambda w$$

**Hessian (Second Derivative):**
$$H_{total} = H_{data} + 2\lambda I$$

Where $H_{data}$ is the Hessian of the data loss.

**Effect:**
- Adds $2\lambda$ to all diagonal elements
- Makes the loss landscape more "convex"
- Improves conditioning (if $H_{data}$ is ill-conditioned)
- Smooths the optimization landscape

**L1 Regularization:**
$$L_{total} = L_{data}(w) + \lambda \sum |w_i|$$

**Subgradient:**
$$\partial L_{total} = \nabla L_{data} + \lambda \cdot \text{sign}(w)$$

**Effect:**
- Non-smooth (not differentiable at zero)
- Creates "kinks" in the loss landscape
- Can create flat regions (when weights are zero)
- Less smooth optimization

**Mathematical Comparison:**

**For a single weight $w$:**

**L2 Penalty:** $P_2(w) = \lambda w^2$
- Derivative: $P_2'(w) = 2\lambda w$ (proportional to weight)
- Second derivative: $P_2''(w) = 2\lambda$ (constant, smooth)

**L1 Penalty:** $P_1(w) = \lambda |w|$
- Subgradient: $P_1'(w) = \lambda \cdot \text{sign}(w)$ (constant magnitude)
- Second derivative: $P_1''(w) = 0$ (except at zero, where undefined)

**Optimization Behavior:**

**L2:**
- Smooth optimization (differentiable everywhere)
- Gradient-based methods work well
- Weights approach zero asymptotically
- Never exactly zero (unless initialized at zero)

**L1:**
- Non-smooth optimization (requires subgradient methods)
- Can get stuck at zero (if data gradient is small)
- Weights can be exactly zero
- Requires special handling (e.g., proximal gradient methods)

**Proximal Gradient Method (for L1):**

For L1, we use proximal gradient descent:

$$w^{k+1} = \text{prox}_{\alpha\lambda\|\cdot\|_1}(w^k - \alpha \nabla L_{data}(w^k))$$

Where the proximal operator is:
$$\text{prox}_{\alpha\lambda\|\cdot\|_1}(w) = \text{sign}(w) \max(|w| - \alpha\lambda, 0)$$

This is the **soft thresholding** operator, which sets small weights to exactly zero!

**Key Insight:**

- **L2:** Smooth penalty, smooth optimization, weights shrink but don't become zero
- **L1:** Non-smooth penalty, requires special methods, weights can become exactly zero (sparsity)

---

## 6. Derive the relationship between regularization and the maximum margin principle in support vector machines. Show how L2 regularization relates to margin maximization.

**Answer:**

**Support Vector Machine (SVM) Formulation:**

The SVM optimization problem is:

$$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to:
$$y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Where:
- $\frac{1}{2}\|w\|^2$ is the L2 regularization term
- $C$ controls the trade-off between margin and misclassification
- $\xi_i$ are slack variables

**Dual Form (Kernel SVM):**

The dual problem maximizes the margin:

$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

Subject to:
$$0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

**Relationship to Regularization:**

**Key Insight:** The $\frac{1}{2}\|w\|^2$ term in SVM is exactly L2 regularization!

**Mathematical Derivation:**

**Margin Definition:**

The margin is the distance from the decision boundary to the nearest data point:

$$\text{margin} = \frac{1}{\|w\|}$$

**Maximizing Margin:**

To maximize the margin, we minimize $\|w\|$, or equivalently, minimize $\frac{1}{2}\|w\|^2$.

**Regularized Loss Formulation:**

The SVM problem can be written as:

$$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))$$

This is:
$$L_{total} = \lambda \|w\|^2 + L_{hinge}(w)$$

Where:
- $\lambda = \frac{1}{2C}$ (regularization strength)
- $L_{hinge}$ is the hinge loss (misclassification penalty)

**Connection to Neural Networks:**

For neural networks with L2 regularization:

$$L_{total} = L_{data}(w) + \lambda \|w\|^2$$

**Similarity:**
- Both minimize weight magnitude
- Both trade off between fitting data and keeping weights small
- Both encourage "simple" solutions

**Difference:**
- SVM: Hard margin maximization (geometric interpretation)
- Neural Networks: Soft regularization (statistical interpretation)

**Mathematical Proof of Margin Maximization:**

**Primal Problem:**
$$\min_{w,b} \frac{1}{2}\|w\|^2$$

Subject to:
$$y_i(w^T x_i + b) \geq 1$$

**Lagrangian:**
$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^{n} \alpha_i[y_i(w^T x_i + b) - 1]$$

**KKT Conditions:**

1. $\frac{\partial L}{\partial w} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0$
   $$\Rightarrow w = \sum_{i=1}^{n} \alpha_i y_i x_i$$

2. $\frac{\partial L}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0$

**Substituting into Lagrangian:**

$$L(\alpha) = \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

**Margin Calculation:**

From KKT condition 1:
$$\|w\|^2 = \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

The margin is:
$$\text{margin} = \frac{1}{\|w\|} = \frac{1}{\sqrt{\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j}}$$

**Key Insight:**

Minimizing $\|w\|^2$ (L2 regularization) is equivalent to maximizing the margin $\frac{1}{\|w\|}$!

**Extension to Neural Networks:**

While neural networks don't have the same geometric margin interpretation, L2 regularization:
- Encourages smaller weights
- Creates smoother decision boundaries
- Improves generalization (similar to large margin)

**Mathematical Relationship:**

For a linear classifier $f(x) = w^T x + b$:

**Margin:** Distance to nearest point
$$\text{margin} = \min_i \frac{y_i(w^T x_i + b)}{\|w\|}$$

**To maximize margin:** Minimize $\|w\|$ (or $\|w\|^2$)

**Regularization:** Minimize $\|w\|^2$

**Conclusion:** L2 regularization in neural networks shares the same mathematical principle as margin maximization in SVMs—both encourage "simple" solutions with small weights.

---

## 7. Explain the mathematical relationship between regularization and the Bayesian interpretation of machine learning. Derive the connection between L2 regularization and Gaussian priors.

**Answer:**

**Bayesian Framework:**

In Bayesian machine learning, we treat model parameters as random variables with prior distributions.

**Maximum A Posteriori (MAP) Estimation:**

Instead of maximum likelihood:
$$\hat{w}_{ML} = \arg\max_w P(\text{data}|w)$$

We use MAP:
$$\hat{w}_{MAP} = \arg\max_w P(w|\text{data}) = \arg\max_w P(\text{data}|w) \cdot P(w)$$

Using Bayes' theorem:
$$P(w|\text{data}) = \frac{P(\text{data}|w) P(w)}{P(\text{data})} \propto P(\text{data}|w) P(w)$$

**Log-Likelihood:**

Taking the logarithm:
$$\log P(w|\text{data}) = \log P(\text{data}|w) + \log P(w) + \text{constant}$$

**MAP Estimation:**
$$\hat{w}_{MAP} = \arg\max_w [\log P(\text{data}|w) + \log P(w)]$$

**Equivalently (minimizing negative log):**
$$\hat{w}_{MAP} = \arg\min_w [-\log P(\text{data}|w) - \log P(w)]$$

**Connection to Regularization:**

This is exactly:
$$\hat{w}_{MAP} = \arg\min_w [L_{data}(w) + R(w)]$$

Where:
- $L_{data}(w) = -\log P(\text{data}|w)$ (negative log-likelihood)
- $R(w) = -\log P(w)$ (negative log-prior)

**L2 Regularization and Gaussian Prior:**

**Gaussian Prior:**

Assume weights have a Gaussian prior:
$$P(w) = \prod_{i=1}^{p} \frac{1}{\sqrt{2\pi\sigma_w^2}} \exp\left(-\frac{w_i^2}{2\sigma_w^2}\right)$$

Where $\sigma_w^2$ is the prior variance.

**Log-Prior:**
$$\log P(w) = \sum_{i=1}^{p} \left[-\frac{1}{2}\log(2\pi\sigma_w^2) - \frac{w_i^2}{2\sigma_w^2}\right]$$

$$= -\frac{p}{2}\log(2\pi\sigma_w^2) - \frac{1}{2\sigma_w^2} \sum_{i=1}^{p} w_i^2$$

**Negative Log-Prior (Regularization Term):**
$$-\log P(w) = \frac{1}{2\sigma_w^2} \sum_{i=1}^{p} w_i^2 + \text{constant}$$

**This is L2 Regularization!**

With $\lambda = \frac{1}{2\sigma_w^2}$:
$$R(w) = \lambda \sum_{i=1}^{p} w_i^2 = \lambda \|w\|^2$$

**Interpretation:**

- **Small $\sigma_w^2$ (tight prior):** Large $\lambda$ → strong regularization → weights constrained to be small
- **Large $\sigma_w^2$ (wide prior):** Small $\lambda$ → weak regularization → weights can be larger

**L1 Regularization and Laplace Prior:**

**Laplace Prior:**

$$P(w) = \prod_{i=1}^{p} \frac{1}{2b} \exp\left(-\frac{|w_i|}{b}\right)$$

**Log-Prior:**
$$\log P(w) = -p\log(2b) - \frac{1}{b} \sum_{i=1}^{p} |w_i|$$

**Negative Log-Prior:**
$$-\log P(w) = \frac{1}{b} \sum_{i=1}^{p} |w_i| + \text{constant}$$

**This is L1 Regularization!**

With $\lambda = \frac{1}{b}$:
$$R(w) = \lambda \sum_{i=1}^{p} |w_i| = \lambda \|w\|_1$$

**Full Bayesian Inference:**

Instead of MAP (point estimate), full Bayesian inference computes the posterior distribution:

$$P(w|\text{data}) = \frac{P(\text{data}|w) P(w)}{P(\text{data})}$$

And makes predictions by integrating over the posterior:
$$P(y_{new}|x_{new}, \text{data}) = \int P(y_{new}|x_{new}, w) P(w|\text{data}) dw$$

**Key Insights:**

1. **L2 Regularization = Gaussian Prior:** Assumes weights are normally distributed around zero
2. **L1 Regularization = Laplace Prior:** Assumes weights have a Laplace distribution (heavier tails, more sparsity)
3. **Regularization Strength = Prior Precision:** $\lambda$ controls how "confident" we are in the prior
4. **Bayesian Interpretation:** Regularization encodes our prior belief that weights should be small

**Mathematical Example:**

For linear regression with Gaussian noise and Gaussian prior:

**Likelihood:**
$$P(y|X, w) = \prod_{i=1}^{n} \mathcal{N}(y_i|w^T x_i, \sigma^2)$$

**Prior:**
$$P(w) = \mathcal{N}(w|0, \sigma_w^2 I)$$

**Posterior:**
$$P(w|y, X) \propto \exp\left(-\frac{1}{2\sigma^2}\|y - Xw\|^2 - \frac{1}{2\sigma_w^2}\|w\|^2\right)$$

**MAP Estimate:**
$$\hat{w}_{MAP} = (X^T X + \frac{\sigma^2}{\sigma_w^2} I)^{-1} X^T y$$

This is exactly ridge regression with $\lambda = \frac{\sigma^2}{\sigma_w^2}$!

---

## 8. Derive the mathematical relationship between regularization and the effective degrees of freedom in a model. Show how L2 regularization reduces model complexity.

**Answer:**

**Degrees of Freedom:**

In statistics, degrees of freedom measure the effective number of parameters in a model.

**Linear Regression (No Regularization):**

For linear regression $\hat{y} = X\hat{\beta}$:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

The degrees of freedom is:
$$\text{df} = \text{rank}(X) = p$$

Where $p$ is the number of parameters.

**Ridge Regression (L2 Regularization):**

For ridge regression:
$$\hat{\beta}_\lambda = (X^T X + \lambda I)^{-1} X^T y$$

**Effective Degrees of Freedom:**

The effective degrees of freedom is:
$$\text{df}(\lambda) = \text{tr}(H_\lambda)$$

Where $H_\lambda$ is the hat matrix:
$$H_\lambda = X(X^T X + \lambda I)^{-1} X^T$$

**Derivation:**

**Hat Matrix:**
$$\hat{y} = X\hat{\beta}_\lambda = X(X^T X + \lambda I)^{-1} X^T y = H_\lambda y$$

**Degrees of Freedom:**
$$\text{df}(\lambda) = \text{tr}(H_\lambda) = \text{tr}(X(X^T X + \lambda I)^{-1} X^T)$$

Using trace property $\text{tr}(AB) = \text{tr}(BA)$:
$$= \text{tr}((X^T X + \lambda I)^{-1} X^T X)$$

**Eigendecomposition:**

Let $X^T X = V \Lambda V^T$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p)$:

$$(X^T X + \lambda I)^{-1} = V(\Lambda + \lambda I)^{-1} V^T$$

$$(X^T X + \lambda I)^{-1} X^T X = V(\Lambda + \lambda I)^{-1} \Lambda V^T$$

**Trace:**
$$\text{df}(\lambda) = \text{tr}(V(\Lambda + \lambda I)^{-1} \Lambda V^T) = \text{tr}((\Lambda + \lambda I)^{-1} \Lambda)$$

$$= \sum_{i=1}^{p} \frac{\lambda_i}{\lambda_i + \lambda}$$

**Key Properties:**

1. **As $\lambda \to 0$:**
   $$\text{df}(\lambda) \to \sum_{i=1}^{p} 1 = p$$
   (Full degrees of freedom, no regularization)

2. **As $\lambda \to \infty$:**
   $$\text{df}(\lambda) \to \sum_{i=1}^{p} 0 = 0$$
   (Zero degrees of freedom, all weights shrink to zero)

3. **For finite $\lambda$:**
   $$0 < \text{df}(\lambda) < p$$
   (Reduced degrees of freedom)

**Interpretation:**

- **Large eigenvalues $\lambda_i$:** Contribute more to degrees of freedom
- **Small eigenvalues $\lambda_i$:** Contribute less (regularized away)
- **Regularization reduces effective complexity:** Fewer "active" parameters

**Neural Networks:**

For neural networks, the effective degrees of freedom is more complex but follows similar principles:

**Approximation:**

For a neural network with L2 regularization:
$$\text{df}(\lambda) \approx \sum_{i=1}^{p} \frac{\lambda_i}{\lambda_i + \lambda}$$

Where $\lambda_i$ are eigenvalues of the Fisher information matrix (or Hessian approximation).

**Model Complexity Reduction:**

**Without Regularization:**
- Full model capacity
- All parameters can vary freely
- High degrees of freedom
- Can overfit

**With L2 Regularization:**
- Reduced effective capacity
- Parameters constrained (weights smaller)
- Lower degrees of freedom
- Less prone to overfitting

**Mathematical Example:**

For a model with eigenvalues $\lambda = [10, 5, 2, 1, 0.1]$:

**No Regularization ($\lambda = 0$):**
$$\text{df} = 5$$ (all parameters active)

**Light Regularization ($\lambda = 0.1$):**
$$\text{df} = \frac{10}{10.1} + \frac{5}{5.1} + \frac{2}{2.1} + \frac{1}{1.1} + \frac{0.1}{0.2} \approx 4.5$$

**Moderate Regularization ($\lambda = 1$):**
$$\text{df} = \frac{10}{11} + \frac{5}{6} + \frac{2}{3} + \frac{1}{2} + \frac{0.1}{1.1} \approx 3.2$$

**Strong Regularization ($\lambda = 10$):**
$$\text{df} = \frac{10}{20} + \frac{5}{15} + \frac{2}{12} + \frac{1}{11} + \frac{0.1}{10.1} \approx 0.8$$

**Key Insight:**

Regularization reduces the effective number of parameters, making the model simpler and less prone to overfitting. The reduction depends on both the regularization strength and the "importance" of each parameter (eigenvalue).

---

## 9. Explain the mathematical relationship between regularization and the condition number of the optimization problem. Show how L2 regularization improves numerical stability.

**Answer:**

**Condition Number:**

The condition number of a matrix $A$ measures how sensitive the solution of $Ax = b$ is to changes in $b$:

$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

Where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values.

**Linear Regression Problem:**

For linear regression:
$$X^T X \beta = X^T y$$

The condition number is:
$$\kappa(X^T X) = \frac{\lambda_{\max}(X^T X)}{\lambda_{\min}(X^T X)}$$

**Problems with Ill-Conditioning:**

1. **Large condition number:** Small changes in $y$ cause large changes in $\hat{\beta}$
2. **Numerical instability:** Rounding errors are amplified
3. **Slow convergence:** Optimization algorithms converge slowly
4. **Overfitting risk:** Model is sensitive to noise

**Ridge Regression (L2 Regularization):**

For ridge regression:
$$(X^T X + \lambda I) \beta = X^T y$$

**Condition Number:**
$$\kappa(X^T X + \lambda I) = \frac{\lambda_{\max}(X^T X) + \lambda}{\lambda_{\min}(X^T X) + \lambda}$$

**Improvement:**

**Without Regularization:**
- If $\lambda_{\min}(X^T X) \approx 0$ (near-singular): $\kappa \to \infty$
- Very ill-conditioned

**With Regularization:**
- $\lambda_{\min}(X^T X) + \lambda \geq \lambda > 0$
- Condition number is bounded: $\kappa \leq \frac{\lambda_{\max} + \lambda}{\lambda}$

**Mathematical Proof:**

**Eigendecomposition:**

Let $X^T X = V \Lambda V^T$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p)$:

$$X^T X + \lambda I = V(\Lambda + \lambda I) V^T$$

**Eigenvalues:**
$$\lambda_i(X^T X + \lambda I) = \lambda_i(X^T X) + \lambda$$

**Condition Number:**
$$\kappa(X^T X + \lambda I) = \frac{\max_i(\lambda_i + \lambda)}{\min_i(\lambda_i + \lambda)} = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$$

**Improvement Ratio:**

$$\frac{\kappa(X^T X)}{\kappa(X^T X + \lambda I)} = \frac{\lambda_{\max}/\lambda_{\min}}{(\lambda_{\max} + \lambda)/(\lambda_{\min} + \lambda)} = \frac{\lambda_{\max}(\lambda_{\min} + \lambda)}{\lambda_{\min}(\lambda_{\max} + \lambda)}$$

If $\lambda_{\min} \ll \lambda$:
$$\approx \frac{\lambda_{\max} \lambda}{\lambda_{\min} \lambda_{\max}} = \frac{\lambda}{\lambda_{\min}}$$

**Example:**

For $X^T X$ with eigenvalues $[100, 10, 1, 0.01]$:

**Without Regularization:**
$$\kappa = \frac{100}{0.01} = 10,000$$ (very ill-conditioned!)

**With Regularization ($\lambda = 1$):**
$$\kappa = \frac{100 + 1}{0.01 + 1} = \frac{101}{1.01} \approx 100$$ (much better!)

**Improvement:** $10,000 / 100 = 100\times$ better conditioning!

**Neural Networks:**

For neural networks, the Hessian matrix $H$ (second derivative of loss) determines conditioning:

**Hessian:**
$$H = \frac{\partial^2 L}{\partial w^2}$$

**With L2 Regularization:**
$$H_{total} = H_{data} + 2\lambda I$$

**Eigenvalues:**
$$\lambda_i(H_{total}) = \lambda_i(H_{data}) + 2\lambda$$

**Condition Number:**
$$\kappa(H_{total}) = \frac{\lambda_{\max}(H_{data}) + 2\lambda}{\lambda_{\min}(H_{data}) + 2\lambda}$$

**Benefits:**

1. **Bounded minimum eigenvalue:** $\lambda_{\min} + 2\lambda \geq 2\lambda > 0$
2. **Better conditioning:** Especially if $H_{data}$ is ill-conditioned
3. **Faster convergence:** Optimization algorithms work better
4. **Numerical stability:** Less sensitive to rounding errors

**Optimization Convergence:**

**Gradient Descent Convergence Rate:**

The convergence rate depends on the condition number:

$$\|w^{k+1} - w^*\| \leq \left(1 - \frac{2\alpha}{\lambda_{\max} + \lambda_{\min}}\right) \|w^k - w^*\|$$

For ill-conditioned problems ($\lambda_{\max} \gg \lambda_{\min}$), convergence is slow.

**With Regularization:**

The effective condition number improves, leading to faster convergence.

**Mathematical Example:**

**Without Regularization:**
- $\lambda_{\max} = 100$, $\lambda_{\min} = 0.01$
- Condition number: $10,000$
- Convergence factor: $\approx 0.98$ (very slow)

**With Regularization ($\lambda = 1$):**
- $\lambda_{\max} = 102$, $\lambda_{\min} = 2.01$
- Condition number: $\approx 50$
- Convergence factor: $\approx 0.04$ (much faster!)

**Key Insight:**

L2 regularization improves the condition number by:
1. Adding a positive constant to all eigenvalues
2. Bounding the minimum eigenvalue away from zero
3. Reducing the ratio of max to min eigenvalues
4. Making optimization more stable and faster

---

## 10. Derive the mathematical relationship between cross-validation and regularization strength selection. Show how to find the optimal lambda using cross-validation.

**Answer:**

**Cross-Validation Framework:**

K-fold cross-validation splits data into $K$ folds, trains on $K-1$ folds, and validates on the remaining fold.

**Mathematical Formulation:**

For $K$ folds, let $D_k$ be the $k$-th fold and $D_{-k}$ be all other folds.

**Cross-Validation Score:**

$$CV(\lambda) = \frac{1}{K} \sum_{k=1}^{K} L(\hat{w}_\lambda^{(k)}, D_k)$$

Where:
- $\hat{w}_\lambda^{(k)}$ is the model trained on $D_{-k}$ with regularization $\lambda$
- $L(\hat{w}_\lambda^{(k)}, D_k)$ is the loss on validation fold $D_k$

**Optimal Lambda:**

$$\lambda^* = \arg\min_{\lambda} CV(\lambda)$$

**Ridge Regression Example:**

**Training on $D_{-k}$:**

$$\hat{w}_\lambda^{(k)} = \arg\min_w \left[\sum_{(x_i, y_i) \in D_{-k}} (y_i - w^T x_i)^2 + \lambda \|w\|^2\right]$$

**Closed-Form Solution:**

$$\hat{w}_\lambda^{(k)} = (X_{-k}^T X_{-k} + \lambda I)^{-1} X_{-k}^T y_{-k}$$

**Validation Loss:**

$$L(\hat{w}_\lambda^{(k)}, D_k) = \sum_{(x_i, y_i) \in D_k} (y_i - (\hat{w}_\lambda^{(k)})^T x_i)^2$$

**Cross-Validation Score:**

$$CV(\lambda) = \frac{1}{K} \sum_{k=1}^{K} \sum_{(x_i, y_i) \in D_k} (y_i - (\hat{w}_\lambda^{(k)})^T x_i)^2$$

**Grid Search:**

To find optimal $\lambda$, evaluate $CV(\lambda)$ for a grid of values:

$$\Lambda = \{\lambda_1, \lambda_2, \ldots, \lambda_m\}$$

$$\lambda^* = \arg\min_{\lambda \in \Lambda} CV(\lambda)$$

**Mathematical Properties:**

**Bias of CV Estimate:**

The cross-validation estimate is approximately unbiased:

$$\mathbb{E}[CV(\lambda)] \approx \mathbb{E}[L(\hat{w}_\lambda, D_{test})]$$

Where $D_{test}$ is an independent test set.

**Variance:**

The variance of CV estimate:
$$\text{Var}(CV(\lambda)) = \frac{1}{K^2} \sum_{k=1}^{K} \text{Var}(L(\hat{w}_\lambda^{(k)}, D_k))$$

**Leave-One-Out Cross-Validation (LOOCV):**

Special case where $K = n$ (one sample per fold):

$$CV_{LOO}(\lambda) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_\lambda^{(-i)})^2$$

Where $\hat{y}_\lambda^{(-i)}$ is prediction using model trained on all data except sample $i$.

**Computational Shortcut for Ridge Regression:**

For ridge regression, LOOCV can be computed efficiently:

$$CV_{LOO}(\lambda) = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{1 - H_{ii}}\right)^2$$

Where $H$ is the hat matrix:
$$H = X(X^T X + \lambda I)^{-1} X^T$$

**Generalized Cross-Validation (GCV):**

Approximation that's computationally cheaper:

$$GCV(\lambda) = \frac{1}{n} \frac{\|y - \hat{y}_\lambda\|^2}{(1 - \text{df}(\lambda)/n)^2}$$

Where $\text{df}(\lambda)$ is the effective degrees of freedom.

**Algorithm for Finding Optimal Lambda:**

1. **Define grid:** $\Lambda = \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$

2. **For each $\lambda \in \Lambda$:**
   - For $k = 1$ to $K$:
     - Train model on $D_{-k}$ with regularization $\lambda$
     - Evaluate on $D_k$
   - Compute $CV(\lambda)$

3. **Select optimal:**
   $$\lambda^* = \arg\min_{\lambda \in \Lambda} CV(\lambda)$$

4. **Retrain on full data:**
   $$\hat{w}^* = \arg\min_w [L_{data}(w) + \lambda^* \|w\|^2]$$

**Mathematical Example:**

For 5-fold CV with $\lambda \in \{0.0001, 0.001, 0.01, 0.1\}$:

| $\lambda$ | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | $CV(\lambda)$ |
|-----------|--------|--------|--------|--------|--------|---------------|
| 0.0001    | 0.52   | 0.48   | 0.51   | 0.49   | 0.50   | 0.50          |
| 0.001     | 0.35   | 0.33   | 0.34   | 0.32   | 0.36   | 0.34          |
| 0.01      | 0.28   | 0.27   | 0.29   | 0.26   | 0.28   | 0.276         |
| 0.1       | 0.45   | 0.43   | 0.44   | 0.42   | 0.46   | 0.44          |

**Optimal:** $\lambda^* = 0.01$ (lowest CV score: 0.276)

**Key Insights:**

1. **Cross-validation estimates generalization error:** $CV(\lambda)$ approximates test error
2. **Optimal lambda minimizes CV score:** Balances bias and variance
3. **Grid search:** Systematically explores regularization space
4. **Computational cost:** $K \times |\Lambda|$ model trainings

**Bias-Variance Decomposition of CV:**

$$CV(\lambda) = \text{Bias}^2(\lambda) + \text{Var}(\lambda) + \text{Irreducible Error}$$

Cross-validation finds $\lambda$ that minimizes this total error, automatically balancing bias and variance!

---











