# Day 2 - Easy Interview Questions

## 1. What is a perceptron?

**Answer:**
A perceptron is the simplest form of an artificial neuron, introduced by Frank Rosenblatt in 1957. It's a linear classifier that makes binary decisions (outputs 0 or 1). The perceptron takes multiple inputs, multiplies each by a corresponding weight, sums them with a bias term, and passes the result through a step function to produce a binary output.

---

## 2. What are the main components of a perceptron?

**Answer:**
The main components are:
1. **Inputs (x₁, x₂, ..., xₙ)**: The input features
2. **Weights (w₁, w₂, ..., wₙ)**: Parameters that represent the importance of each input
3. **Bias (b)**: A threshold parameter that shifts the decision boundary
4. **Weighted Sum (z)**: Calculated as `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
5. **Activation Function**: A step function that converts the weighted sum to binary output (0 or 1)

---

## 3. What is the mathematical formula for a perceptron's output?

**Answer:**
The perceptron computes:
1. **Weighted Sum**: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b` (or `z = w·x + b` in vector notation)
2. **Activation**: 
   ```
   output = 1 if z ≥ 0
   output = 0 if z < 0
   ```

---

## 4. What is the purpose of weights in a perceptron?

**Answer:**
Weights represent the **importance** or **strength** of each input feature. A larger weight means that input has more influence on the final decision. Weights are learned during training to minimize prediction errors.

---

## 5. What is the purpose of bias in a perceptron?

**Answer:**
The bias acts as a **threshold** that shifts the decision boundary left or right. It allows the perceptron to make better-informed decisions by adjusting when the weighted sum crosses zero. Without bias, the decision boundary would always pass through the origin, limiting the perceptron's flexibility.

---

## 6. What is the Perceptron Learning Rule?

**Answer:**
The Perceptron Learning Rule is how the perceptron updates its weights and bias during training:
1. Make a prediction using current weights and bias
2. Calculate the error: `error = target - prediction`
3. Update weights: `w_new = w_old + (learning_rate × error × input)`
4. Update bias: `b_new = b_old + (learning_rate × error)`

The learning rate controls the step size of these updates.

---

## 7. What is a decision boundary?

**Answer:**
A decision boundary is the line (in 2D) or hyperplane (in higher dimensions) that separates different classes. For a perceptron, it's the line where `w₁x₁ + w₂x₂ + b = 0`. Points on one side are classified as 0, and points on the other side are classified as 1.

---

## 8. What does "linearly separable" mean?

**Answer:**
A problem is **linearly separable** if a single straight line (or hyperplane in higher dimensions) can perfectly separate the two classes. For example, the OR gate is linearly separable because you can draw a line separating the (0,0) point from the other three points.

---

## 9. What is the XOR problem and why can't a perceptron solve it?

**Answer:**
The XOR (exclusive OR) problem has the following truth table:
- (0, 0) → 0
- (0, 1) → 1
- (1, 0) → 1
- (1, 1) → 0

A perceptron **cannot** solve XOR because it's not linearly separable - you cannot draw a single straight line to separate the 0s and 1s. The points (0,1) and (1,0) are on one side, while (0,0) and (1,1) are on the other, forming a pattern that requires a curved boundary.

---

## 10. What is the Perceptron Convergence Theorem?

**Answer:**
The Perceptron Convergence Theorem states that:
- **If a dataset is linearly separable**, the perceptron learning algorithm is **guaranteed** to find a separating line in a finite number of steps.
- **If a dataset is NOT linearly separable** (like XOR), the algorithm will **never converge** and will loop forever, endlessly updating its weights.

---

## 11. What is a step function (Heaviside function)?

**Answer:**
A step function (also called Heaviside step function) is the activation function used in perceptrons. It outputs:
- `1` if the input is greater than or equal to 0
- `0` if the input is less than 0

This hard-limiter function forces the perceptron to make a definitive binary choice.

---

## 12. What is the learning rate in the context of perceptron training?

**Answer:**
The learning rate is a hyperparameter that controls the step size of weight updates during training. It's typically a small positive value (e.g., 0.01 or 0.1). 
- **Too high**: May overshoot optimal weights and fail to converge
- **Too low**: Very slow convergence, takes many iterations
- **Just right**: Stable and efficient learning

---

## 13. How does a perceptron make a prediction?

**Answer:**
To make a prediction:
1. Calculate the weighted sum: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
2. Apply the step function: `output = 1 if z ≥ 0, else 0`
3. Return the binary output (0 or 1)

This is called the **forward pass**.

---

## 14. What happens when a perceptron makes a correct prediction?

**Answer:**
When the prediction is correct:
- The error is 0: `error = target - prediction = 0`
- According to the update rule: `w_new = w_old + (learning_rate × 0 × input) = w_old`
- **No change** is made to the weights or bias

---

## 15. Why is the perceptron important in the history of neural networks?

**Answer:**
The perceptron is important because:
1. It's the **simplest neural network** and foundation for understanding complex networks
2. It introduces core concepts: weights, biases, activation functions, and learning rules
3. Its **limitation** (cannot solve XOR) directly led to the development of **Multi-Layer Perceptrons (MLPs)**
4. All modern deep learning concepts build upon these fundamental ideas

---

## 16. What is the difference between training and inference for a perceptron?

**Answer:**
- **Training**: The process of learning optimal weights and bias by iteratively updating them based on prediction errors. Requires both forward pass (prediction) and backward pass (weight updates).

- **Inference**: Using the trained perceptron to make predictions on new data. Only requires forward pass (no weight updates).

---

## 17. Can a perceptron solve the AND gate problem?

**Answer:**
Yes! The AND gate is linearly separable:
- (0, 0) → 0
- (0, 1) → 0
- (1, 0) → 0
- (1, 1) → 1

You can draw a line separating the single (1,1) point from the other three points, so a perceptron can learn to solve it.

---

## 18. What happens if you initialize all weights to zero in a perceptron?

**Answer:**
If all weights start at zero:
- The weighted sum will always be just the bias: `z = 0·x + b = b`
- The perceptron will initially predict the same class for all inputs (depending on bias sign)
- It can still learn, but may take longer to converge
- It's a common initialization strategy, though small random values are often preferred

---

## 19. What is the relationship between perceptron and linear regression?

**Answer:**
Both use linear combinations of inputs:
- **Linear Regression**: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b` (continuous output)
- **Perceptron**: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`, then `output = step(z)` (binary output)

The perceptron is essentially linear regression followed by a step function for binary classification.

---

## 20. How many parameters does a perceptron with n inputs have?

**Answer:**
A perceptron with n inputs has:
- **n weights** (one for each input)
- **1 bias** (single threshold parameter)
- **Total: n + 1 parameters**

For example, a perceptron with 2 inputs has 3 parameters (2 weights + 1 bias).

