# Day 7 - Medium Interview Questions

## 1. Derive the mathematical formula for gradient descent and explain the convergence conditions.

**Answer:**

**Gradient Descent Formula:**

Starting with the loss function $L(w)$, we want to minimize it by updating weights:

$$w_{t+1} = w_t - \alpha \cdot \nabla_w L(w_t)$$

Where:
- $w_t$ is the weight at iteration $t$
- $\alpha$ is the learning rate
- $\nabla_w L(w_t)$ is the gradient of the loss with respect to weights

**Derivation:**

Using Taylor expansion around $w_t$:
$$L(w_{t+1}) \approx L(w_t) + \nabla_w L(w_t)^T (w_{t+1} - w_t)$$

To minimize $L(w_{t+1})$, we want:
$$\nabla_w L(w_t)^T (w_{t+1} - w_t) < 0$$

This is negative when:
$$w_{t+1} - w_t = -\alpha \cdot \nabla_w L(w_t)$$

Therefore:
$$w_{t+1} = w_t - \alpha \cdot \nabla_w L(w_t)$$

**Convergence Conditions:**

**1. Learning Rate Bound:**
For convergence, learning rate must satisfy:
$$0 < \alpha < \frac{2}{\lambda_{\max}}$$

Where $\lambda_{\max}$ is the maximum eigenvalue of the Hessian matrix.

**2. Lipschitz Continuity:**
The gradient must be Lipschitz continuous:
$$||\nabla L(w_1) - \nabla L(w_2)|| \leq L ||w_1 - w_2||$$

**3. Convexity (for Global Minimum):**
For convex functions, gradient descent converges to global minimum.

**4. Sufficient Decrease Condition:**
$$L(w_{t+1}) \leq L(w_t) - c \alpha ||\nabla L(w_t)||^2$$

Where $c > 0$ is a constant.

**Convergence Rate:**

For strongly convex functions with Lipschitz gradient:
$$L(w_t) - L(w^*) \leq (1 - \mu \alpha)^t (L(w_0) - L(w^*))$$

Where $\mu$ is the strong convexity parameter.

---

## 2. Derive the momentum update rule and explain why it accelerates convergence.

**Answer:**

**Momentum Update Rule:**

Momentum maintains a velocity vector that accumulates gradients:

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla_w L(w_t)$$

$$w_{t+1} = w_t - \alpha \cdot v_t$$

Where:
- $v_t$ is the velocity at iteration $t$
- $\beta$ is the momentum coefficient (typically 0.9)
- $\alpha$ is the learning rate

**Alternative Formulation (Common in Practice):**

$$v_t = \beta \cdot v_{t-1} + \nabla_w L(w_t)$$

$$w_{t+1} = w_t - \alpha \cdot v_t$$

**Derivation:**

**Intuition:** We want to use information from past gradients to smooth the update.

**Velocity Accumulation:**
The velocity is an exponentially weighted moving average of gradients:

$$v_t = \beta v_{t-1} + \nabla L_t$$
$$= \beta(\beta v_{t-2} + \nabla L_{t-1}) + \nabla L_t$$
$$= \beta^2 v_{t-2} + \beta \nabla L_{t-1} + \nabla L_t$$
$$= \sum_{i=0}^{t} \beta^{t-i} \nabla L_i$$

**Why It Accelerates:**

**1. Smoothing Effect:**
- Averages past gradients, reducing noise
- Smoother optimization path
- Less oscillation

**2. Direction Persistence:**
- If gradients consistently point in same direction, velocity builds up
- Larger effective step size in consistent directions
- Faster convergence

**3. Escape from Local Minima:**
- Accumulated velocity can carry optimizer over small bumps
- Helps escape shallow local minima

**Mathematical Analysis:**

**Convergence Rate:**
For quadratic functions, momentum can improve convergence from $O(1/t)$ to $O((1-\sqrt{\mu/L})^t)$ where $\mu$ and $L$ are strong convexity and Lipschitz constants.

**Effective Learning Rate:**
The effective step size in direction of consistent gradients is approximately:
$$\alpha_{\text{eff}} \approx \frac{\alpha}{1 - \beta}$$

For $\beta = 0.9$ and $\alpha = 0.01$:
$$\alpha_{\text{eff}} \approx \frac{0.01}{0.1} = 0.1$$

This is 10x larger than without momentum!

**Key Insight:**
Momentum accelerates convergence by building up speed in consistent directions while smoothing out noise, effectively increasing the learning rate in the right directions.

---

## 3. Derive Nesterov Accelerated Gradient and explain why it's better than standard momentum.

**Answer:**

**Nesterov Accelerated Gradient (NAG):**

NAG computes the gradient at a "look-ahead" position:

$$v_t = \beta \cdot v_{t-1} + \nabla_w L(w_t - \beta \cdot v_{t-1})$$

$$w_{t+1} = w_t - \alpha \cdot v_t$$

**Key Difference:**
The gradient is computed at $(w_t - \beta \cdot v_{t-1})$, which is where we would be after taking a momentum step.

**Derivation:**

**Standard Momentum:**
1. Compute gradient at current position: $\nabla L(w_t)$
2. Update velocity: $v_t = \beta v_{t-1} + \nabla L(w_t)$
3. Update position: $w_{t+1} = w_t - \alpha v_t$

**Nesterov Momentum:**
1. Look ahead: Compute where momentum would take us: $w_t - \beta v_{t-1}$
2. Compute gradient at look-ahead position: $\nabla L(w_t - \beta v_{t-1})$
3. Update velocity: $v_t = \beta v_{t-1} + \nabla L(w_t - \beta v_{t-1})$
4. Update position: $w_{t+1} = w_t - \alpha v_t$

**Why It's Better:**

**1. Corrective Behavior:**
- By looking ahead, NAG can "see" if momentum is about to overshoot
- Can correct the direction before overshooting
- More accurate updates

**2. Better Convergence:**
- Typically converges faster than standard momentum
- Especially near the minimum where overshooting is a problem
- Better final accuracy

**3. Mathematical Guarantee:**
- For convex functions, NAG has better convergence rate
- Convergence rate: $O(1/t^2)$ vs. $O(1/t)$ for standard momentum

**Convergence Analysis:**

For strongly convex functions:
- **Standard momentum:** $O((1-\sqrt{\mu/L})^t)$
- **Nesterov:** $O((1-\sqrt[4]{\mu/L})^t)$

The fourth root is larger than the square root, so Nesterov converges faster.

**Intuitive Explanation:**

**Standard Momentum:**
- Like running downhill with momentum
- You might overshoot turns because you're committed to your momentum direction

**Nesterov:**
- Like running downhill but constantly looking ahead
- You can see where your momentum will take you
- You can adjust your path before overshooting

**Mathematical Example:**

**Scenario:** Approaching a minimum with momentum

**Standard Momentum:**
- Current position: $w_t = 5$
- Momentum direction: moving left (negative)
- Gradient at $w_t$: also negative
- Update: continues left, might overshoot minimum

**Nesterov:**
- Current position: $w_t = 5$
- Look-ahead position: $w_t - \beta v_{t-1} = 4$ (where momentum takes us)
- Gradient at look-ahead: might be positive (past minimum!)
- Update: corrects direction, reduces overshooting

**Key Insight:**
NAG is better because it uses gradient information from where it's going, not just where it is, allowing it to correct overshooting before it happens.

---

## 4. Derive the RMSprop update rule and explain how it adapts learning rates per parameter.

**Answer:**

**RMSprop Update Rule:**

RMSprop maintains a running average of squared gradients:

$$E[g^2]_t = \beta \cdot E[g^2]_{t-1} + (1 - \beta) \cdot g_t^2$$

$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

Where:
- $E[g^2]_t$ is the running average of squared gradients
- $\beta$ is the decay rate (typically 0.9)
- $g_t$ is the current gradient
- $\epsilon$ is a small constant for numerical stability (typically 1e-8)
- $\alpha$ is the base learning rate

**Derivation:**

**Problem:** Different parameters have gradients of very different magnitudes.

**Solution:** Normalize the learning rate by the magnitude of recent gradients.

**Squared Gradient Average:**
The running average $E[g^2]_t$ estimates the second moment (variance) of gradients:

$$E[g^2]_t = (1-\beta) \sum_{i=0}^{t} \beta^{t-i} g_i^2$$

This is an exponentially weighted moving average.

**Adaptive Learning Rate:**
The effective learning rate for parameter $i$ is:
$$\alpha_{\text{eff}}^{(i)} = \frac{\alpha}{\sqrt{E[g^2]_t^{(i)} + \epsilon}}$$

**How It Adapts:**

**For Parameters with Large Gradients:**
- $E[g^2]_t$ is large
- $\sqrt{E[g^2]_t}$ is large
- $\alpha_{\text{eff}} = \frac{\alpha}{\text{large}} = \text{small}$
- **Result:** Smaller learning rate (prevents overshooting)

**For Parameters with Small Gradients:**
- $E[g^2]_t$ is small
- $\sqrt{E[g^2]_t}$ is small
- $\alpha_{\text{eff}} = \frac{\alpha}{\text{small}} = \text{large}$
- **Result:** Larger learning rate (faster learning)

**Mathematical Properties:**

**1. Per-Parameter Adaptation:**
Each parameter gets its own learning rate based on its gradient history.

**2. Handles Non-Stationary Objectives:**
The running average adapts to changing gradient magnitudes.

**3. Numerical Stability:**
The $\epsilon$ term prevents division by zero when $E[g^2]_t = 0$.

**Comparison to Fixed Learning Rate:**

**Fixed Learning Rate:**
- All parameters use same learning rate
- Parameters with large gradients might overshoot
- Parameters with small gradients might learn slowly

**RMSprop:**
- Each parameter gets appropriate learning rate
- Large gradients → small steps
- Small gradients → large steps
- More efficient optimization

**Key Insight:**
RMSprop adapts learning rates per parameter by normalizing by the magnitude of recent gradients, allowing efficient optimization when different parameters have very different gradient scales.

---

## 5. Derive the Adam optimizer update rule and explain how it combines momentum and adaptive learning rates.

**Answer:**

**Adam Update Rule:**

Adam maintains two running averages:

**First Moment (Mean of Gradients):**
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

**Second Moment (Mean of Squared Gradients):**
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**
$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

Where:
- $m_t$ is the first moment estimate
- $v_t$ is the second moment estimate
- $\beta_1$ is the first moment decay (typically 0.9)
- $\beta_2$ is the second moment decay (typically 0.999)
- $\alpha$ is the learning rate
- $\epsilon$ is a small constant (typically 1e-8)

**Derivation:**

**1. First Moment (Momentum):**
The first moment $m_t$ is an exponentially weighted moving average of gradients:
$$m_t = (1-\beta_1) \sum_{i=0}^{t} \beta_1^{t-i} g_i$$

This provides momentum-like behavior, smoothing gradients.

**2. Second Moment (RMSprop-like):**
The second moment $v_t$ is an exponentially weighted moving average of squared gradients:
$$v_t = (1-\beta_2) \sum_{i=0}^{t} \beta_2^{t-i} g_i^2$$

This estimates the variance of gradients, used for adaptive learning rates.

**3. Bias Correction:**
Early in training, $m_t$ and $v_t$ are biased toward zero (since they start at 0).

**Unbiased Estimates:**
$$\mathbb{E}[m_t] = (1-\beta_1^t) \mathbb{E}[g_t]$$

So the unbiased estimate is:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

Similarly:
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**4. Combined Update:**
The update combines:
- **Momentum term:** $\hat{m}_t$ (smoothed gradient direction)
- **Adaptive learning rate:** $\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}$ (per-parameter scaling)

**Why It Works:**

**1. Combines Best of Both:**
- **Momentum from $m_t$:** Smooths gradients, accelerates in consistent directions
- **Adaptive rates from $v_t$:** Adjusts learning rate per parameter

**2. Handles Sparse Gradients:**
- Second moment adapts to gradient magnitude
- Works well when gradients are sparse or noisy

**3. Automatic Tuning:**
- Works well with default hyperparameters
- Less sensitive to learning rate choice

**Mathematical Properties:**

**Convergence:**
For convex functions, Adam converges to optimal solution with rate $O(1/\sqrt{t})$.

**Memory:**
- Stores two moving averages per parameter
- Memory efficient compared to storing full gradient history

**Adaptation:**
- Adapts both gradient direction (via $m_t$) and step size (via $v_t$)
- More sophisticated than either momentum or RMSprop alone

**Key Insight:**
Adam combines the smoothing of momentum ($m_t$) with the adaptive learning rates of RMSprop ($v_t$), while correcting for bias in early iterations. This makes it both fast (momentum) and adaptive (per-parameter rates).

---

## 6. Compare the convergence rates of SGD, SGD with momentum, and Adam. Provide mathematical analysis.

**Answer:**

**Convergence Rate Analysis:**

**1. SGD (Stochastic Gradient Descent):**

For strongly convex functions with Lipschitz gradient:

**Deterministic (Full Batch):**
$$L(w_t) - L(w^*) \leq (1 - \mu \alpha)^t (L(w_0) - L(w^*))$$

Convergence rate: $O((1-\mu\alpha)^t)$

**Stochastic (Mini-Batch):**
$$\mathbb{E}[L(w_t) - L(w^*)] \leq \frac{C}{\sqrt{t}}$$

Convergence rate: $O(1/\sqrt{t})$

Where $C$ depends on gradient variance.

**2. SGD with Momentum:**

For strongly convex functions:

**Deterministic:**
$$L(w_t) - L(w^*) \leq O\left(\left(1 - \sqrt{\frac{\mu}{L}}\right)^t\right)$$

Convergence rate: $O((1-\sqrt{\mu/L})^t)$

**Comparison to SGD:**
- If $\mu \ll L$ (ill-conditioned), momentum is much faster
- Example: If $\mu/L = 0.01$, then:
  - SGD: $(1-0.01\alpha)^t$
  - Momentum: $(1-0.1)^t$ (10x faster!)

**Stochastic:**
$$\mathbb{E}[L(w_t) - L(w^*)] \leq O\left(\frac{C}{\sqrt{t}}\right)$$

Similar to SGD, but with smaller constant $C$ due to variance reduction.

**3. Adam:**

For convex functions:

$$\mathbb{E}[L(w_t) - L(w^*)] \leq O\left(\frac{1}{\sqrt{t}}\right)$$

Convergence rate: $O(1/\sqrt{t})$

**However:**
- Adam typically has better constants in practice
- Adapts to gradient scale, often faster in practice
- Better for non-convex problems

**Mathematical Comparison:**

**For Well-Conditioned Problems ($\mu \approx L$):**
- **SGD:** $O((1-c)^t)$ - Fast
- **Momentum:** $O((1-c)^t)$ - Similar
- **Adam:** $O(1/\sqrt{t})$ - Slower theoretically, but often faster in practice

**For Ill-Conditioned Problems ($\mu \ll L$):**
- **SGD:** $O((1-\mu\alpha)^t)$ - Very slow
- **Momentum:** $O((1-\sqrt{\mu/L})^t)$ - Much faster
- **Adam:** $O(1/\sqrt{t})$ - Adapts, often best

**Practical Considerations:**

**SGD:**
- Simple, guaranteed convergence
- Can be slow for ill-conditioned problems
- Requires careful learning rate tuning

**Momentum:**
- Faster for ill-conditioned problems
- Still requires learning rate tuning
- Better than SGD in most cases

**Adam:**
- Adapts automatically
- Often fastest in practice
- Less sensitive to hyperparameters
- Better for non-convex problems

**Key Insight:**
While theoretical convergence rates vary, in practice:
- **Momentum** is typically faster than SGD (especially for ill-conditioned problems)
- **Adam** often performs best in practice due to adaptation, even if theoretical rate is $O(1/\sqrt{t})$

---

## 7. Explain the relationship between learning rate and batch size. Derive the linear scaling rule.

**Answer:**

**The Relationship:**

There's a relationship between batch size and optimal learning rate:
- **Larger batch size** → More stable gradients → Can use larger learning rate
- **Smaller batch size** → Noisier gradients → Need smaller learning rate

**Linear Scaling Rule:**

If you multiply batch size by $k$, you can roughly multiply learning rate by $k$:

$$\alpha_{\text{new}} = k \cdot \alpha_{\text{old}}$$

Where batch size is also multiplied by $k$.

**Derivation:**

**Gradient Estimate:**

For batch size $b$, the gradient estimate is:
$$\hat{g}_b = \frac{1}{b} \sum_{i=1}^{b} \nabla L_i$$

**Variance of Gradient Estimate:**

The variance of the gradient estimate decreases with batch size:
$$\text{Var}(\hat{g}_b) = \frac{\text{Var}(\nabla L)}{b}$$

**For Batch Size $b$:**
- Gradient estimate: $\hat{g}_b = \frac{1}{b} \sum_{i=1}^{b} \nabla L_i$
- Variance: $\text{Var}(\hat{g}_b) = \frac{\sigma^2}{b}$

**For Batch Size $kb$:**
- Gradient estimate: $\hat{g}_{kb} = \frac{1}{kb} \sum_{i=1}^{kb} \nabla L_i$
- Variance: $\text{Var}(\hat{g}_{kb}) = \frac{\sigma^2}{kb} = \frac{1}{k} \cdot \frac{\sigma^2}{b}$

**Effective Step Size:**

The effective step size should be proportional to the signal-to-noise ratio.

**For Batch Size $b$:**
- Signal: $\mathbb{E}[\hat{g}_b] = \nabla L$ (unbiased)
- Noise: $\sqrt{\text{Var}(\hat{g}_b)} = \frac{\sigma}{\sqrt{b}}$
- Signal-to-noise: $\frac{|\nabla L| \sqrt{b}}{\sigma}$

**For Batch Size $kb$:**
- Signal: $\mathbb{E}[\hat{g}_{kb}] = \nabla L$ (same)
- Noise: $\sqrt{\text{Var}(\hat{g}_{kb})} = \frac{\sigma}{\sqrt{kb}} = \frac{1}{\sqrt{k}} \cdot \frac{\sigma}{\sqrt{b}}$
- Signal-to-noise: $\frac{|\nabla L| \sqrt{kb}}{\sigma} = \sqrt{k} \cdot \frac{|\nabla L| \sqrt{b}}{\sigma}$

**To Maintain Same Signal-to-Noise Ratio:**

If we want the same effective step size, we need:
$$\alpha_{kb} \cdot |\hat{g}_{kb}| = \alpha_b \cdot |\hat{g}_b|$$

Since $|\hat{g}_{kb}| \approx |\hat{g}_b|$ (same expected magnitude), but variance is $1/k$:
$$\alpha_{kb} \approx k \cdot \alpha_b$$

**Limitations:**

**1. Not Exact:**
- Linear scaling is a rough guideline
- Actual relationship is more complex
- Depends on problem and optimizer

**2. Very Large Batches:**
- For very large batch sizes, linear scaling may not hold
- Diminishing returns
- May need different scaling (e.g., square root)

**3. Optimizer Dependent:**
- Works better for SGD
- For adaptive optimizers (Adam), relationship is different
- Adam already adapts, so scaling may be less important

**Practical Guidelines:**

**For SGD:**
- Batch size 32, lr=0.01 → Batch size 64, lr=0.02
- Batch size 32, lr=0.01 → Batch size 128, lr=0.04

**For Adam:**
- Less critical (already adaptive)
- Can try linear scaling, but may not need it
- Often works fine with same learning rate

**Key Insight:**
Larger batches provide more stable gradient estimates with lower variance, allowing larger learning rates. The linear scaling rule ($\alpha \propto b$) is a useful guideline, especially for SGD, though it's approximate and may not hold for very large batches or adaptive optimizers.

---

## 8. Derive the relationship between learning rate and the condition number of the Hessian matrix. Explain how this affects optimization.

**Answer:**

**Condition Number:**

The condition number of the Hessian matrix $H$ is:
$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

Where $\lambda_{\max}$ and $\lambda_{\min}$ are the maximum and minimum eigenvalues.

**Learning Rate Bound:**

For gradient descent to converge, the learning rate must satisfy:
$$0 < \alpha < \frac{2}{\lambda_{\max}(H)}$$

**Optimal Learning Rate:**

For quadratic functions, the optimal learning rate is:
$$\alpha^* = \frac{2}{\lambda_{\max}(H) + \lambda_{\min}(H)}$$

**Convergence Rate:**

With optimal learning rate, convergence rate is:
$$L(w_t) - L(w^*) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^t (L(w_0) - L(w^*))$$

Where $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ is the condition number.

**Derivation:**

**Quadratic Approximation:**

Near a minimum, the loss can be approximated as:
$$L(w) \approx L(w^*) + \frac{1}{2}(w - w^*)^T H (w - w^*)$$

**Gradient:**
$$\nabla L(w) = H(w - w^*)$$

**Gradient Descent Update:**
$$w_{t+1} = w_t - \alpha H(w_t - w^*)$$

**Error:**
$$e_{t+1} = w_{t+1} - w^* = w_t - w^* - \alpha H(w_t - w^*) = (I - \alpha H) e_t$$

**Eigenvalue Decomposition:**

If $H$ has eigenvalues $\lambda_i$ with eigenvectors $v_i$:
$$e_{t+1} = (I - \alpha H) e_t = \sum_i (1 - \alpha \lambda_i) (e_t^T v_i) v_i$$

**Convergence Condition:**

For convergence, we need $|1 - \alpha \lambda_i| < 1$ for all $i$:
$$-1 < 1 - \alpha \lambda_i < 1$$
$$0 < \alpha \lambda_i < 2$$
$$0 < \alpha < \frac{2}{\lambda_i}$$

For all eigenvalues, we need:
$$0 < \alpha < \frac{2}{\lambda_{\max}}$$

**Convergence Rate:**

The slowest convergence is for the eigenvalue closest to the bound:
$$\rho = \max_i |1 - \alpha \lambda_i|$$

For optimal $\alpha = \frac{2}{\lambda_{\max} + \lambda_{\min}}$:
$$\rho = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\kappa - 1}{\kappa + 1}$$

**Effect of Condition Number:**

**Well-Conditioned ($\kappa \approx 1$):**
- $\lambda_{\max} \approx \lambda_{\min}$
- $\rho \approx 0$ (fast convergence)
- Learning rate can be large
- Easy to optimize

**Ill-Conditioned ($\kappa \gg 1$):**
- $\lambda_{\max} \gg \lambda_{\min}$
- $\rho \approx 1$ (slow convergence)
- Learning rate must be small (bounded by $\lambda_{\max}$)
- Hard to optimize

**Example:**

**Well-Conditioned:**
- $\lambda_{\max} = 10$, $\lambda_{\min} = 8$, $\kappa = 1.25$
- Optimal $\alpha = \frac{2}{18} = 0.111$
- Convergence rate: $\rho = \frac{1.25-1}{1.25+1} = 0.111$ (fast!)

**Ill-Conditioned:**
- $\lambda_{\max} = 1000$, $\lambda_{\min} = 1$, $\kappa = 1000$
- Optimal $\alpha = \frac{2}{1001} = 0.002$
- Convergence rate: $\rho = \frac{1000-1}{1000+1} = 0.998$ (very slow!)

**How Momentum Helps:**

Momentum improves convergence for ill-conditioned problems:
- Convergence rate: $O((1-\sqrt{\mu/L})^t)$ instead of $O((1-\mu/L)^t)$
- For $\kappa = 1000$: $\sqrt{\mu/L} = \sqrt{1/1000} = 0.032$ vs. $\mu/L = 0.001$
- Much faster!

**Key Insight:**
The condition number $\kappa$ determines how difficult optimization is. Ill-conditioned problems ($\kappa \gg 1$) require small learning rates and converge slowly. Momentum and adaptive optimizers help by improving convergence rates for ill-conditioned problems.

---

## 9. Explain the mathematical relationship between optimizers and the loss landscape. How do different optimizers navigate different types of landscapes?

**Answer:**

**Loss Landscape:**

The loss function $L(w)$ creates a landscape in parameter space. The shape of this landscape determines optimization difficulty.

**Types of Landscapes:**

**1. Convex (Bowl-shaped):**
- Single global minimum
- No local minima
- Easy to optimize

**2. Non-Convex (Many Local Minima):**
- Multiple local minima
- Hard to find global minimum
- Common in deep learning

**3. Ill-Conditioned (Narrow Valleys):**
- Very different curvatures in different directions
- Long, narrow valleys
- Hard to navigate

**4. Saddle Points:**
- Points where gradient is zero but not a minimum
- Common in high-dimensional spaces
- Can trap optimizers

**How Optimizers Navigate:**

**1. SGD:**
- Follows gradient direction
- Can get stuck in local minima
- Oscillates in narrow valleys
- Can escape saddle points (due to noise)

**Mathematical Behavior:**
$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

- Moves in direction of steepest descent
- Step size constant
- No memory of past gradients

**2. SGD with Momentum:**
- Builds up velocity in consistent directions
- Can escape shallow local minima
- Smoother path in narrow valleys
- May overshoot

**Mathematical Behavior:**
$$v_t = \beta v_{t-1} + \nabla L(w_t)$$
$$w_{t+1} = w_t - \alpha v_t$$

- Velocity accumulates, providing inertia
- Can roll over small bumps
- Smoother optimization path

**3. Adam:**
- Adapts learning rate per parameter
- Combines momentum with adaptive rates
- Good for various landscape types

**Mathematical Behavior:**
$$w_{t+1} = w_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot m_t$$

- Adapts to local curvature
- Different step sizes in different directions
- Can navigate complex landscapes

**Landscape-Specific Behavior:**

**Convex Landscapes:**
- **SGD:** Works well, converges to global minimum
- **Momentum:** Faster convergence
- **Adam:** Also works, may be overkill

**Narrow Valleys (Ill-Conditioned):**
- **SGD:** Oscillates, slow convergence
- **Momentum:** Smoother, faster
- **Adam:** Adapts step sizes, often best

**Many Local Minima:**
- **SGD:** Can get stuck
- **Momentum:** Can escape shallow minima
- **Adam:** Noise helps escape, adaptive rates help

**Saddle Points:**
- **SGD:** Can get stuck (gradient is zero)
- **Momentum:** Velocity can carry past
- **Adam:** Adaptive rates help navigate

**Mathematical Analysis:**

**Eigenvalue Structure:**

For a point in parameter space, the Hessian $H$ has eigenvalues $\lambda_i$.

**SGD:**
- Step size: $\alpha$ (same in all directions)
- May be too large for large $\lambda_i$, too small for small $\lambda_i$

**Momentum:**
- Effective step size varies with direction
- Better for ill-conditioned problems

**Adam:**
- Step size: $\frac{\alpha}{\sqrt{v_t} + \epsilon}$ (adapts per parameter)
- Automatically adjusts to local curvature
- Optimal for varying eigenvalue structure

**Key Insight:**
Different optimizers navigate loss landscapes differently:
- **SGD:** Simple, works for simple landscapes
- **Momentum:** Better for narrow valleys and escaping local minima
- **Adam:** Adapts to complex landscapes with varying curvature

The choice of optimizer should match the complexity of your loss landscape.

---

## 10. Derive the relationship between learning rate schedules and convergence. Explain different scheduling strategies mathematically.

**Answer:**

**Learning Rate Schedule:**

A learning rate schedule changes $\alpha$ over time: $\alpha(t)$

**Fixed Learning Rate:**
$$\alpha(t) = \alpha_0 \quad \text{(constant)}$$

**Why Schedules Help:**

**Early Training:**
- Model far from optimal
- Large learning rate → fast initial learning

**Late Training:**
- Model close to optimal
- Small learning rate → fine-tuning, avoid overshooting

**Common Schedules:**

**1. Step Decay:**

$$\alpha(t) = \alpha_0 \cdot \gamma^{\lfloor t/T \rfloor}$$

Where:
- $\alpha_0$ is initial learning rate
- $\gamma$ is decay factor (typically 0.1)
- $T$ is decay period
- $\lfloor t/T \rfloor$ is number of decay steps

**Example:**
- $\alpha_0 = 0.1$, $\gamma = 0.1$, $T = 30$
- Epochs 0-29: $\alpha = 0.1$
- Epochs 30-59: $\alpha = 0.01$
- Epochs 60-89: $\alpha = 0.001$

**2. Exponential Decay:**

$$\alpha(t) = \alpha_0 \cdot e^{-kt}$$

Where $k$ is the decay rate.

**Properties:**
- Smooth, continuous decay
- Never reaches zero (asymptotically)

**3. Polynomial Decay:**

$$\alpha(t) = \alpha_0 \cdot \left(1 - \frac{t}{T}\right)^p$$

Where $T$ is total epochs and $p$ is power (typically 1 or 2).

**4. Cosine Annealing:**

$$\alpha(t) = \alpha_{\min} + (\alpha_0 - \alpha_{\min}) \cdot \frac{1 + \cos(\pi t / T)}{2}$$

Where:
- $\alpha_0$ is initial learning rate
- $\alpha_{\min}$ is minimum learning rate
- $T$ is period

**Properties:**
- Smooth, periodic
- Starts at $\alpha_0$, ends at $\alpha_{\min}$

**5. Reduce on Plateau:**

$$\alpha(t) = \begin{cases}
\alpha(t-1) & \text{if improvement} \\
\gamma \cdot \alpha(t-1) & \text{if no improvement for } N \text{ epochs}
\end{cases}$$

Adaptive: only reduces when validation loss stops improving.

**Convergence Analysis:**

**Fixed Learning Rate:**

For strongly convex functions:
$$L(w_t) - L(w^*) \leq (1 - \mu \alpha)^t (L(w_0) - L(w^*))$$

Converges if $\alpha < 2/\lambda_{\max}$.

**Decaying Learning Rate:**

For $\alpha(t) = \frac{c}{t}$:
$$\mathbb{E}[L(w_t) - L(w^*)] \leq O\left(\frac{1}{t}\right)$$

Better than fixed rate for stochastic optimization.

**Optimal Schedule:**

For strongly convex functions, optimal schedule is:
$$\alpha(t) = \frac{1}{\mu t}$$

This gives convergence rate $O(1/t)$.

**Why Schedules Work:**

**1. Exploration vs. Exploitation:**
- Early: Large LR → explore, find good region
- Late: Small LR → exploit, fine-tune

**2. Avoid Overshooting:**
- Near minimum, large steps overshoot
- Small steps allow precise convergence

**3. Escape Local Minima:**
- Large LR early can escape poor local minima
- Small LR late ensures convergence

**Mathematical Example:**

**Fixed LR = 0.01:**
- May overshoot near minimum
- Oscillates around optimum

**Decaying LR:**
- Starts at 0.01, decays to 0.001
- Fast initial learning
- Precise final convergence

**Key Insight:**
Learning rate schedules improve convergence by:
- Using large rates early (fast learning)
- Using small rates late (precise convergence)
- Adapting to training progress

The optimal schedule depends on the problem, but decaying schedules generally outperform fixed rates.

---

