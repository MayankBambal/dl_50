# Day 12 - Medium Interview Questions

## 1. Derive the relationship between learning rate and convergence rate. Show mathematically how learning rate affects optimization speed and stability.

**Answer:**

**Convergence Rate:**

For gradient descent, the convergence rate depends on the learning rate and the condition number of the Hessian.

**Gradient Descent Update:**
$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

Where $\alpha$ is the learning rate.

**Convergence Analysis:**

For a strongly convex function with condition number $\kappa = \frac{L}{\mu}$ (where $L$ is Lipschitz constant, $\mu$ is strong convexity parameter), the convergence rate is:

$$\|w_t - w^*\|^2 \leq (1 - \frac{2\alpha\mu}{1+\alpha L})^t \|w_0 - w^*\|^2$$

**Optimal Learning Rate:**

The optimal learning rate that maximizes convergence rate is:
$$\alpha^* = \frac{2}{L + \mu}$$

**Convergence Rate:**
$$\|w_t - w^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|w_0 - w^*\|^2$$

**Effect of Learning Rate:**

**Too Small ($\alpha << \alpha^*$):**
- Convergence is slow: $(1 - \frac{2\alpha\mu}{1+\alpha L}) \approx 1$
- Takes many iterations to converge
- Stable but inefficient

**Too Large ($\alpha > \frac{2}{L}$):**
- May diverge: $(1 - \frac{2\alpha\mu}{1+\alpha L}) > 1$
- Training becomes unstable
- Loss may explode

**Optimal ($\alpha \approx \alpha^*$):**
- Fast convergence: $(1 - \frac{2\alpha\mu}{1+\alpha L})$ minimized
- Stable training
- Efficient optimization

**Stability Condition:**

For stability, we need:
$$|1 - \alpha L| < 1$$

Which gives:
$$0 < \alpha < \frac{2}{L}$$

**Key Insight:** Learning rate must be small enough for stability but large enough for fast convergence. The optimal learning rate balances these trade-offs.

---

## 2. Derive the relationship between batch size and gradient variance. Show how batch size affects optimization dynamics.

**Answer:**

**Gradient with Batch Size:**

For a batch of size $B$, the gradient is:
$$\hat{g}_B = \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(w)$$

Where $L_i$ is the loss for example $i$.

**True Gradient:**
$$g = \mathbb{E}[\nabla L_i(w)] = \frac{1}{N} \sum_{i=1}^{N} \nabla L_i(w)$$

**Gradient Variance:**

The variance of the batch gradient is:
$$\text{Var}(\hat{g}_B) = \text{Var}\left(\frac{1}{B} \sum_{i=1}^{B} \nabla L_i(w)\right)$$

Assuming independent samples:
$$\text{Var}(\hat{g}_B) = \frac{1}{B^2} \sum_{i=1}^{B} \text{Var}(\nabla L_i(w)) = \frac{1}{B} \text{Var}(\nabla L_i(w))$$

**Key Relationship:**
$$\text{Var}(\hat{g}_B) = \frac{\sigma^2}{B}$$

Where $\sigma^2 = \text{Var}(\nabla L_i(w))$ is the variance of individual gradients.

**Effect of Batch Size:**

**Small Batch ($B$ small):**
- High variance: $\text{Var}(\hat{g}_B) = \frac{\sigma^2}{B}$ (large)
- Noisy gradients
- More exploration
- May help escape local minima

**Large Batch ($B$ large):**
- Low variance: $\text{Var}(\hat{g}_B) = \frac{\sigma^2}{B}$ (small)
- Stable gradients
- Less exploration
- Faster convergence per epoch

**Optimization Dynamics:**

The weight update with batch gradient:
$$w_{t+1} = w_t - \alpha \hat{g}_B$$

The expected update:
$$\mathbb{E}[w_{t+1}] = w_t - \alpha g$$

The variance in updates:
$$\text{Var}(w_{t+1}) = \alpha^2 \text{Var}(\hat{g}_B) = \frac{\alpha^2 \sigma^2}{B}$$

**Trade-off:**

- **Small batch:** High variance (noise) can help generalization but slows convergence
- **Large batch:** Low variance (stable) speeds convergence but may hurt generalization

**Effective Learning Rate:**

Some research suggests scaling learning rate with batch size:
$$\alpha_{effective} = \alpha \cdot \frac{B}{B_{ref}}$$

Where $B_{ref}$ is a reference batch size (e.g., 32).

**Key Insight:** Batch size controls gradient variance. Smaller batches = more noise = potentially better generalization but slower training. Larger batches = less noise = faster training but potentially worse generalization.

---

## 3. Derive the relationship between learning rate and batch size from an optimization perspective. Show when and why you might scale learning rate with batch size.

**Answer:**

**Gradient Variance Relationship:**

As derived in question 2:
$$\text{Var}(\hat{g}_B) = \frac{\sigma^2}{B}$$

**Weight Update Variance:**

The variance in weight updates:
$$\text{Var}(\Delta w) = \alpha^2 \text{Var}(\hat{g}_B) = \frac{\alpha^2 \sigma^2}{B}$$

**Constant Update Variance:**

To maintain constant variance in updates when changing batch size:
$$\text{Var}(\Delta w) = \text{constant}$$

This requires:
$$\frac{\alpha^2 \sigma^2}{B} = \text{constant}$$

Which gives:
$$\alpha \propto \sqrt{B}$$

**Linear Scaling Rule:**

However, a common heuristic is **linear scaling**:
$$\alpha(B) = \alpha_{ref} \cdot \frac{B}{B_{ref}}$$

Where:
- $\alpha_{ref}$ is learning rate for reference batch size $B_{ref}$
- $B$ is the new batch size

**Why Linear Scaling:**

For SGD, if we want the same expected update magnitude:
$$\mathbb{E}[\Delta w] = -\alpha \cdot \mathbb{E}[\hat{g}_B] = -\alpha \cdot g$$

To maintain the same expected update when batch size changes:
$$\alpha(B) \cdot g = \alpha_{ref} \cdot g$$

This suggests:
$$\alpha(B) = \alpha_{ref}$$

But this ignores variance effects.

**Square Root Scaling:**

For constant variance:
$$\alpha(B) = \alpha_{ref} \cdot \sqrt{\frac{B}{B_{ref}}}$$

**When to Scale:**

**For SGD:**
- Linear scaling often works: $\alpha(B) = \alpha_{ref} \cdot \frac{B}{B_{ref}}$
- Especially for large batch sizes

**For Adam/RMSprop:**
- Relationship is weaker (adaptive learning rates)
- May not need scaling
- Default learning rate often works across batch sizes

**Practical Approach:**

```python
# Reference: batch_size=32, lr=0.001
B_ref = 32
alpha_ref = 0.001

# New batch size
B_new = 128

# Linear scaling
alpha_new = alpha_ref * (B_new / B_ref)  # 0.001 * 4 = 0.004

# Or square root scaling
alpha_new = alpha_ref * np.sqrt(B_new / B_ref)  # 0.001 * 2 = 0.002
```

**Key Insight:** For SGD with large batch sizes, linear scaling ($\alpha \propto B$) often works. For adaptive optimizers, scaling is less critical. Always validate with experiments.

---

## 4. Derive the relationship between hyperparameter search space size and search efficiency. Compare grid search and random search complexity.

**Answer:**

**Search Space Size:**

For $d$ hyperparameters with $n_i$ values each:
$$\text{Search Space Size} = \prod_{i=1}^{d} n_i$$

**Grid Search Complexity:**

Grid search tries all combinations:
$$N_{grid} = \prod_{i=1}^{d} n_i$$

**Random Search Complexity:**

Random search tries $N_{random}$ random samples (typically much less than grid search).

**Example:**

For 3 hyperparameters with 3 values each:
- **Grid search:** $3^3 = 27$ combinations
- **Random search:** $N_{random} = 20$ samples (can explore more values)

**Efficiency Comparison:**

**Grid Search:**
- **Coverage:** Systematic, tries all combinations
- **Efficiency:** Exponentially expensive: $O(\prod_{i=1}^{d} n_i)$
- **Best for:** Few hyperparameters ($d \leq 3$)

**Random Search:**
- **Coverage:** Random, explores search space
- **Efficiency:** Linear in number of trials: $O(N_{random})$
- **Best for:** Many hyperparameters ($d > 3$)

**Mathematical Analysis:**

**Grid Search:**
- Must try all $N_{grid} = \prod_{i=1}^{d} n_i$ combinations
- Exponentially grows with $d$

**Random Search:**
- Tries $N_{random}$ random samples
- Can find good solutions with $N_{random} << N_{grid}$

**Why Random Search is Often Better:**

Not all hyperparameters are equally important. Random search can:
- Find good values for important hyperparameters
- Explore more values per hyperparameter
- Use fewer total trials

**Example:**

For 5 hyperparameters with 3 values each:
- **Grid search:** $3^5 = 243$ combinations (must try all)
- **Random search:** $50$ random samples (can explore $50$ different values per hyperparameter)

**Expected Performance:**

For random search with $N$ trials, probability of finding a configuration in top $p$ percentile:
$$P(\text{find top } p\%) = 1 - (1-p)^N$$

For $N=50$ and $p=0.1$ (top 10%):
$$P = 1 - 0.9^{50} \approx 0.995$$

**Key Insight:** Random search is often more efficient than grid search, especially with many hyperparameters. The efficiency advantage grows exponentially with the number of hyperparameters.

---

## 5. Derive the relationship between model capacity and generalization error. Show how hyperparameters affect the bias-variance tradeoff.

**Answer:**

**Generalization Error Decomposition:**

The expected prediction error can be decomposed as:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \text{Irreducible Error}$$

Where:
- **Bias:** $\text{Bias}(\hat{f}) = \mathbb{E}[\hat{f}] - f$ (systematic error)
- **Variance:** $\text{Var}(\hat{f}) = \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2]$ (sensitivity to data)

**Model Capacity:**

Model capacity measures the model's ability to fit complex functions. Higher capacity = can fit more complex functions.

**Effect of Capacity on Bias and Variance:**

**Low Capacity (Small Model):**
- **High Bias:** Model too simple, can't fit data well
- **Low Variance:** Model stable, less sensitive to data
- **Total Error:** High (bias dominates)

**High Capacity (Large Model):**
- **Low Bias:** Model can fit data well
- **High Variance:** Model sensitive to data, overfits
- **Total Error:** High (variance dominates)

**Optimal Capacity:**
- **Balanced Bias and Variance:** Model fits data but doesn't overfit
- **Total Error:** Minimized

**Hyperparameters and Capacity:**

**Network Architecture (Depth/Width):**
- **Larger architecture:** Higher capacity
- **Effect:** Reduces bias, increases variance

**Regularization (Weight Decay, Dropout):**
- **More regularization:** Lower effective capacity
- **Effect:** Increases bias, reduces variance

**Learning Rate:**
- **Too high:** Unstable training, high variance
- **Too low:** Slow convergence, may not reach optimal bias
- **Optimal:** Balances convergence and stability

**Mathematical Model:**

For a model with capacity $C$:
$$\text{Bias}^2(C) \propto \frac{1}{C}$$
$$\text{Var}(C) \propto C$$

Total error:
$$\text{Error}(C) = \frac{a}{C} + bC + c$$

Where $a$, $b$, $c$ are constants.

**Optimal Capacity:**

Minimizing error:
$$\frac{d}{dC} \text{Error}(C) = -\frac{a}{C^2} + b = 0$$

Gives:
$$C^* = \sqrt{\frac{a}{b}}$$

**Hyperparameter Tuning Goal:**

Find hyperparameters that minimize:
$$\min_{\theta} [\text{Bias}^2(\theta) + \text{Var}(\theta)]$$

Where $\theta$ represents hyperparameters.

**Key Insight:** Hyperparameter tuning is about finding the optimal balance between bias and variance. Different hyperparameters affect this balance differently, and the optimal combination minimizes total generalization error.

---

## 6. Derive the relationship between learning rate schedule and convergence. Show mathematically how learning rate decay affects optimization.

**Answer:**

**Learning Rate Schedule:**

A learning rate schedule changes the learning rate over time:
$$\alpha_t = \alpha_0 \cdot \eta(t)$$

Where $\eta(t)$ is the schedule function.

**Common Schedules:**

**1. Step Decay:**
$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Where:
- $\gamma$ is decay factor (e.g., 0.1)
- $s$ is step size (e.g., every 10 epochs)

**2. Exponential Decay:**
$$\alpha_t = \alpha_0 \cdot \gamma^t$$

**3. Polynomial Decay:**
$$\alpha_t = \alpha_0 \cdot (1 - \frac{t}{T})^p$$

Where $T$ is total epochs, $p$ is power.

**4. Cosine Annealing:**
$$\alpha_t = \alpha_0 \cdot \frac{1 + \cos(\pi t / T)}{2}$$

**Convergence Analysis:**

For gradient descent with learning rate schedule:
$$w_{t+1} = w_t - \alpha_t \nabla L(w_t)$$

**Convergence Rate:**

For a strongly convex function, with decreasing learning rate:
$$\|w_t - w^*\|^2 \leq \frac{C}{\sum_{i=0}^{t} \alpha_i}$$

Where $C$ is a constant.

**Why Decay Helps:**

**Early Training:**
- Large learning rate: Fast initial progress
- Explores search space quickly

**Later Training:**
- Small learning rate: Fine-tunes solution
- Reduces oscillations near optimum
- Better final convergence

**Mathematical Justification:**

For convergence, we need:
$$\sum_{t=0}^{\infty} \alpha_t = \infty \quad \text{(exploration)}$$
$$\sum_{t=0}^{\infty} \alpha_t^2 < \infty \quad \text{(convergence)}$$

**Step Decay Example:**

For step decay with $\gamma = 0.1$, $s = 10$:
- Epochs 0-9: $\alpha = \alpha_0$
- Epochs 10-19: $\alpha = 0.1 \alpha_0$
- Epochs 20-29: $\alpha = 0.01 \alpha_0$
- etc.

**Effect on Optimization:**

**Without Decay:**
- Learning rate constant
- May oscillate near optimum
- Slower fine-tuning

**With Decay:**
- Learning rate decreases
- Smoother convergence
- Better final solution

**Key Insight:** Learning rate decay helps optimization by allowing fast initial progress (large LR) followed by fine-tuning (small LR). The schedule should satisfy conditions for both exploration and convergence.

---

## 7. Derive the relationship between hyperparameter sensitivity and search strategy. Show how to identify which hyperparameters are most important to tune.

**Answer:**

**Hyperparameter Sensitivity:**

The sensitivity of a hyperparameter measures how much the objective function changes when the hyperparameter changes:
$$S_i = \frac{\partial L_{val}}{\partial \theta_i}$$

Where $\theta_i$ is hyperparameter $i$.

**Empirical Sensitivity:**

For discrete hyperparameters:
$$S_i \approx \frac{L_{val}(\theta_i + \Delta\theta_i) - L_{val}(\theta_i)}{\Delta\theta_i}$$

**Hyperparameter Importance:**

The importance of hyperparameter $i$ can be measured by:
$$I_i = \max_{\theta_i} L_{val}(\theta_i) - \min_{\theta_i} L_{val}(\theta_i)$$

Or normalized:
$$I_i = \frac{\max_{\theta_i} L_{val}(\theta_i) - \min_{\theta_i} L_{val}(\theta_i)}{\text{range}(L_{val})}$$

**Search Strategy Based on Sensitivity:**

**High Sensitivity Hyperparameters:**
- Must tune carefully
- Use fine-grained search
- More important to get right

**Low Sensitivity Hyperparameters:**
- Can use coarse search
- Less critical
- Can use defaults

**Example Analysis:**

```python
# Measure sensitivity
sensitivities = {}

# Learning rate
lr_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
lr_losses = [evaluate(lr=l) for l in lr_values]
sensitivities['lr'] = max(lr_losses) - min(lr_losses)  # High!

# Batch size
bs_values = [32, 64, 128, 256]
bs_losses = [evaluate(bs=b) for b in bs_values]
sensitivities['bs'] = max(bs_losses) - min(bs_losses)  # Lower

# Dropout rate
dr_values = [0.3, 0.5, 0.7]
dr_losses = [evaluate(dr=d) for d in dr_values]
sensitivities['dr'] = max(dr_losses) - min(dr_losses)  # Lower still
```

**Optimal Search Allocation:**

Allocate more search budget to high-sensitivity hyperparameters:
$$N_i \propto S_i$$

Where $N_i$ is number of trials for hyperparameter $i$.

**Adaptive Search:**

**Phase 1: Screening**
- Quick evaluation of all hyperparameters
- Identify high-sensitivity ones

**Phase 2: Focused Search**
- Fine-tune high-sensitivity hyperparameters
- Coarse search for low-sensitivity ones

**Mathematical Framework:**

For a search budget $N$, allocate:
$$N_i = N \cdot \frac{S_i}{\sum_j S_j}$$

**Key Insight:** Identify high-sensitivity hyperparameters through screening, then allocate more search budget to them. This makes hyperparameter tuning more efficient.

---

## 8. Derive the relationship between hyperparameter tuning and overfitting to the validation set. Show how to prevent validation set overfitting.

**Answer:**

**Validation Set Overfitting:**

When tuning hyperparameters, you're essentially "training" on the validation set by choosing hyperparameters that perform well on it. This can lead to overfitting to the validation set.

**The Problem:**

After trying many hyperparameter configurations:
$$\theta^* = \arg\min_{\theta} L_{val}(\theta)$$

The selected hyperparameters $\theta^*$ may be overfitted to the validation set, leading to:
$$L_{test}(\theta^*) > L_{val}(\theta^*)$$

**Mathematical Model:**

**True Generalization Error:**
$$L_{gen}(\theta) = \mathbb{E}_{(x,y) \sim P} [L(f(x; \theta), y)]$$

**Validation Error:**
$$L_{val}(\theta) = \frac{1}{N_{val}} \sum_{i=1}^{N_{val}} L(f(x_i; \theta), y_i)$$

**Bias in Validation Error:**

The validation error is an estimate:
$$L_{val}(\theta) = L_{gen}(\theta) + \epsilon_{val}(\theta)$$

Where $\epsilon_{val}(\theta)$ is estimation error.

**Overfitting Effect:**

When selecting:
$$\theta^* = \arg\min_{\theta} L_{val}(\theta)$$

We're selecting:
$$\theta^* = \arg\min_{\theta} [L_{gen}(\theta) + \epsilon_{val}(\theta)]$$

This biases selection toward configurations with negative $\epsilon_{val}(\theta)$ (lucky on validation set).

**Expected Overfitting:**

The expected overfitting is:
$$\mathbb{E}[L_{test}(\theta^*) - L_{val}(\theta^*)] = \mathbb{E}[\epsilon_{test}(\theta^*) - \epsilon_{val}(\theta^*)]$$

**Prevention Strategies:**

**1. Larger Validation Set:**
- Reduces $\epsilon_{val}(\theta)$ variance
- More reliable estimates
- Less overfitting

**2. Cross-Validation:**
- Use $k$-fold cross-validation
- Average over multiple folds
- More robust estimates

**3. Nested Cross-Validation:**
- Outer loop: Final evaluation
- Inner loop: Hyperparameter tuning
- Prevents overfitting

**4. Holdout Test Set:**
- Never use test set for tuning
- Only for final evaluation
- Unbiased estimate

**5. Limit Number of Trials:**
- Fewer hyperparameter configurations tried
- Less opportunity to overfit
- More conservative search

**Mathematical Bound:**

For $K$ hyperparameter configurations tried:
$$\mathbb{E}[L_{test}(\theta^*) - L_{val}(\theta^*)] \leq \sqrt{\frac{2\log K}{N_{val}}}$$

This shows overfitting increases with:
- More configurations tried ($K$)
- Smaller validation set ($N_{val}$)

**Key Insight:** Validation set overfitting is real and increases with more hyperparameter configurations tried. Use larger validation sets, cross-validation, and limit the number of trials to prevent it.

---

## 9. Derive the relationship between hyperparameter search and computational budget. Show how to optimize search strategy given limited compute.

**Answer:**

**Computational Budget:**

Total compute available:
$$B_{total} = \text{time} \times \text{resources}$$

**Search Cost:**

Cost per hyperparameter configuration:
$$C_{config} = T_{train} \times C_{unit}$$

Where:
- $T_{train}$ is training time per config
- $C_{unit}$ is cost per unit time

**Number of Configurations:**

Given budget $B$:
$$N_{max} = \frac{B}{C_{config}}$$

**Search Strategy Optimization:**

Maximize expected performance given budget:
$$\max_{N, \text{strategy}} \mathbb{E}[L_{val}(\theta^*(N))]$$

Subject to:
$$N \cdot C_{config} \leq B$$

**Efficient Strategies:**

**1. Two-Phase Search:**
- **Phase 1:** Quick screening ($T_{quick} << T_{full}$)
  - Train for few epochs
  - Evaluate many configurations
  - Keep top $K$ candidates
- **Phase 2:** Full evaluation
  - Train top $K$ candidates fully
  - Choose best

**Budget Allocation:**
$$B = N_1 \cdot T_{quick} + K \cdot T_{full}$$

Where $N_1$ is number of quick trials, $K$ is number of full trials.

**Optimal Allocation:**

Maximize:
$$\max_{N_1, K} \mathbb{E}[\text{performance}]$$

Subject to:
$$N_1 \cdot T_{quick} + K \cdot T_{full} \leq B$$

**2. Early Stopping:**
- Stop training when not improving
- Saves compute per configuration
- More configurations can be tried

**3. Random Search over Grid Search:**
- More efficient exploration
- Better use of budget
- Finds good solutions faster

**4. Hyperparameter Importance:**
- Focus budget on important hyperparameters
- Less budget on less important ones
- More efficient overall

**Mathematical Optimization:**

For two-phase search:
- Screen $N_1$ configs with $T_{quick}$ each
- Evaluate top $K$ with $T_{full}$ each

Budget constraint:
$$N_1 \cdot T_{quick} + K \cdot T_{full} \leq B$$

Optimal $K$:
$$K^* = \arg\max_K \mathbb{E}[\text{performance}(K)]$$

Subject to budget.

**Example:**

Given:
- $B = 100$ hours
- $T_{quick} = 0.5$ hours (1 epoch)
- $T_{full} = 10$ hours (20 epochs)

**Option 1: All Full**
- $N = 10$ configurations
- All trained fully

**Option 2: Two-Phase**
- Screen 100 configs (1 epoch each) = 50 hours
- Evaluate top 5 fully = 50 hours
- Total: 100 hours
- Better: Can find good configs from larger search space

**Key Insight:** Optimize search strategy for limited compute by using two-phase search (quick screening + full evaluation), early stopping, and focusing on important hyperparameters.

---

## 10. Derive the relationship between hyperparameter tuning and the bias-variance decomposition of the generalization error. Show how different hyperparameters affect bias and variance.

**Answer:**

**Bias-Variance Decomposition:**

For a model with hyperparameters $\theta$:
$$\mathbb{E}[(y - \hat{f}(x; \theta))^2] = \text{Bias}^2(\theta) + \text{Var}(\theta) + \text{Irreducible Error}$$

Where:
- **Bias:** $\text{Bias}(\theta) = \mathbb{E}[\hat{f}(x; \theta)] - f(x)$
- **Variance:** $\text{Var}(\theta) = \mathbb{E}[(\hat{f}(x; \theta) - \mathbb{E}[\hat{f}(x; \theta)])^2]$

**Effect of Hyperparameters:**

**1. Learning Rate:**

**Too High:**
- Unstable training
- High variance (oscillations)
- May not converge (high bias too)

**Too Low:**
- Slow convergence
- May get stuck (high bias)
- Stable (low variance)

**Optimal:**
- Fast convergence (low bias)
- Stable (low variance)

**2. Model Capacity (Architecture):**

**Small Model:**
- High bias (can't fit data)
- Low variance (stable)

**Large Model:**
- Low bias (can fit data)
- High variance (overfits)

**Optimal:**
- Balanced bias and variance

**3. Regularization (Weight Decay, Dropout):**

**No Regularization:**
- Low bias
- High variance (overfitting)

**Too Much Regularization:**
- High bias (underfitting)
- Low variance

**Optimal:**
- Slightly higher bias
- Much lower variance
- Net: Lower total error

**Mathematical Model:**

For model capacity $C$ and regularization strength $\lambda$:
$$\text{Bias}^2(C, \lambda) = \frac{a}{C} + b\lambda$$
$$\text{Var}(C, \lambda) = cC - d\lambda$$

Total error:
$$\text{Error}(C, \lambda) = \frac{a}{C} + b\lambda + cC - d\lambda + e$$

**Optimal Hyperparameters:**

Minimize:
$$\min_{C, \lambda} \text{Error}(C, \lambda)$$

Gives:
$$\frac{\partial \text{Error}}{\partial C} = -\frac{a}{C^2} + c = 0 \Rightarrow C^* = \sqrt{\frac{a}{c}}$$
$$\frac{\partial \text{Error}}{\partial \lambda} = b - d = 0 \Rightarrow \lambda^* = \frac{d}{b}$$

**Hyperparameter Tuning Goal:**

Find hyperparameters that minimize:
$$\min_{\theta} [\text{Bias}^2(\theta) + \text{Var}(\theta)]$$

**Trade-offs:**

**Learning Rate:**
- Affects both bias (convergence) and variance (stability)
- Optimal: Balance both

**Architecture:**
- Primarily affects bias (capacity)
- Secondarily affects variance (overfitting)

**Regularization:**
- Primarily affects variance (overfitting)
- Secondarily affects bias (underfitting)

**Key Insight:** Hyperparameter tuning is about finding the optimal balance between bias and variance. Different hyperparameters affect this balance differently, and the optimal combination minimizes total generalization error through the bias-variance tradeoff.

---

## 11. Derive how Bayesian optimization selects the next hyperparameter set using surrogate models and acquisition functions.

**Answer:**
Bayesian optimization treats hyperparameter tuning as a **black-box optimization** problem and sequentially decides which configuration to evaluate next by modeling uncertainty.

**1. Surrogate Model (Posterior Over Objective):**
- Let $f(\theta)$ be the unknown validation loss for hyperparameters $\theta$.
- Assume a prior $p(f)$ (commonly a Gaussian Process with mean $\mu_0(\theta)$ and kernel $k(\theta, \theta')$).
- After $n$ observations $\mathcal{D}_n = \{(\theta_i, y_i)\}_{i=1}^n$ with $y_i = f(\theta_i) + \epsilon$, update the posterior:
  $$
  p(f|\mathcal{D}_n) = \mathcal{GP}(\mu_n(\theta), \sigma_n^2(\theta))
  $$
  where $\mu_n$ and $\sigma_n^2$ are closed-form GP posterior mean/variance.

**2. Acquisition Function (Decision Rule):**
- Define an acquisition function $a(\theta; \mathcal{D}_n)$ that quantifies the utility of evaluating $\theta$ next.
- Common choices:
  - **Expected Improvement (EI):**
    $$
    \text{EI}(\theta) = \mathbb{E}[\max(0, f^\star - f(\theta))]
    = (f^\star - \mu_n(\theta))\Phi(z) + \sigma_n(\theta)\phi(z)
    $$
    where $f^\star$ is best observed value, $z = \frac{f^\star - \mu_n(\theta)}{\sigma_n(\theta)}$, and $\Phi$, $\phi$ are standard normal CDF/PDF.
  - **Upper Confidence Bound (UCB):**
    $$
    \text{UCB}(\theta) = \mu_n(\theta) - \kappa \sigma_n(\theta)
    $$
    (for minimization; larger $\kappa$ encourages exploration).
  - **Probability of Improvement (PI):**
    $$
    \text{PI}(\theta) = \Phi\left(\frac{f^\star - \mu_n(\theta)}{\sigma_n(\theta)}\right)
    $$

**3. Next Hyperparameter Selection:**
$$
\theta_{n+1} = \arg\max_{\theta} a(\theta; \mathcal{D}_n)
$$
Evaluate $f(\theta_{n+1})$, augment $\mathcal{D}_{n+1}$, and repeat.

**4. Exploration vs. Exploitation:**
- $\mu_n(\theta)$ represents exploitation (prefers low predicted loss).
- $\sigma_n(\theta)$ captures uncertainty (exploration).
- Acquisition functions combine both to avoid wasting trials:
  - EI rewards high potential improvement.
  - UCB explicitly balances mean vs. uncertainty via $\kappa$.

**5. Practical Implementation:**
Libraries like Optuna, scikit-optimize, and Hyperopt automate:
- Building/updating the surrogate (e.g., Tree-structured Parzen Estimator or GP).
- Maximizing the acquisition function (often via inner optimization or sampling).
- Supporting pruning: stop trials early if surrogate predicts poor performance.

**Key Insight:** Bayesian optimization iteratively refines a surrogate model of validation loss and uses an acquisition function to choose the most promising hyperparameter configuration, achieving strong results with far fewer trials than grid or random search.

---

