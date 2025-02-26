# NeuronFromScratch
A project to train a single neuron from scratch to learn a linear function. This serves as a foundational step for more complex neural network models.

# 🚧 In progress ...

## Repository Structure

### `> ./LinearRegression/`

| Filename | Description | Link |
|----------|-------------|------|
| `using-covariance-and-variance.ipynb` | Notebook demonstrating linear regression using covariance and variance to find the slope and intercept. | [Read More](./reading/using-covariance-and-variance.md) |




Your question is very insightful—you're diving deep into **learning rate scheduling**, **gradient scaling**, and **optimization dynamics**. Let's break it down step by step.

---

### **1. Why does `b` take longer to converge than `k`?**
The gradient descent update rule is:  
\[
\theta = \theta - \alpha \cdot \frac{\partial J}{\partial \theta}
\]
For `k`, the gradient is:  
\[
\frac{\partial J}{\partial k} = \frac{2}{N} \sum (y_{\text{pred}} - y) x
\]
For `b`, the gradient is:  
\[
\frac{\partial J}{\partial b} = \frac{2}{N} \sum (y_{\text{pred}} - y)
\]

The issue is that **`b` has a smaller gradient than `k`**, because `b` does not get scaled by `x`.  
- When `x` has large values, `k_gradient` will be much larger than `b_gradient`.
- `k` gets updated significantly, while `b` moves much slower.  
- Over time, `b` has even smaller gradients, making it seem **stuck**.

---

### **2. Why does increasing the learning rate (`b_lr *= 1.03`) help?**
When `b_gradient` is small, `b` needs a **larger learning rate** to make meaningful updates.  
By **increasing the learning rate** over time, you compensate for the decreasing `b_gradient`, allowing `b` to converge faster.

However, if you increase `k_lr` and `b_lr` **without control**, `k` might overshoot and oscillate. That’s why tuning this is tricky.

---

### **3. Two possible approaches to solve this**
You experimented with **two learning rate schedules**:
- **(A) Decreasing LR:** Start with a high LR, then decrease.  
  🔴 **Issue**: Slows down `b` even more because `b_gradient` is already small.  
- **(B) Increasing LR:** Start small, then increase.  
  ✅ **Works better** but might overshoot `k`.  

Here’s how you can improve this:

#### **Approach 1: Use separate learning rates for `k` and `b`**
Since `b_gradient` is naturally smaller, assign it a larger initial learning rate:
```python
k_lr = 0.01
b_lr = 0.1  # Larger because b's updates are smaller
```
Now, `b` moves faster without needing 10,000 iterations.

#### **Approach 2: Adaptive learning rate (like Adam)**
Instead of manually increasing the learning rate, use an adaptive method:
```python
k_lr = 0.01 / (1 + 0.01 * epoch)
b_lr = 0.1 / (1 + 0.01 * epoch)
```
This **decreases** learning rate over time, preventing overshooting.

---

### **4. Theoretical justification**
Your idea of **increasing learning rate** to handle small gradients aligns with modern optimizers like **Adam and RMSprop**, which scale the learning rate based on the magnitude of past gradients.

- If **gradient is large**, a **small LR** prevents overshooting.  
- If **gradient is small**, a **large LR** speeds up convergence.  

However, a naive increase in LR can cause instability. That’s why Adam uses **momentum** and **adaptive scaling**.

---

### **5. Summary**
✅ `b` converges slowly because its gradient is smaller.  
✅ Increasing `b_lr` over time helps, but overshooting can happen.  
✅ A better fix is **starting with a larger `b_lr`** or **using adaptive learning rates**.  
✅ The principle behind increasing `b_lr` aligns with **Adam-like optimizers**.
