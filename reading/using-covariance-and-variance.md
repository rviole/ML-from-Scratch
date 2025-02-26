## **🔷 Variance, Covariance, and Linear Regression: Full Summary**  

### **1️⃣ Variance** 📊  
Variance measures how **spread out** the values of a dataset are. It tells us how much the values of a single variable deviate from the mean.  

**Formula:**  

$$
\text{Var}(X) = \frac{\sum (x_i - \bar{x})^2}{n}
$$

### **2️⃣ Covariance**  
Covariance measures the **relationship** between two variables \(X\) and \(Y\). It tells us **how they move together**:  
- **Positive covariance** → when \(X\) increases, \(Y\) also increases (direct relationship).  
- **Negative covariance** → when \(X\) increases, \(Y\) decreases (inverse relationship).  
- **Zero covariance** → no relationship.  

**Formula:**  

$$
\text{Cov}(X, Y) = \frac{\sum (x_i - \bar{x}) (y_i - \bar{y})}{n}
$$

### **3️⃣ How Do We Use Covariance and Variance to Find the Slope?**  
We use the formula:

$$
k = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

### **4️⃣ How Do We Find the Intercept?**  
Once we have the slope \(k\), we find the intercept \(b\):  

$$
b = \bar{y} - k\bar{x}
$$

### **5️⃣ Why Not Just Use "Rise Over Run" on Two Points?**  
You **can** use two points to find the slope:

$$
k = \frac{y_2 - y_1}{x_2 - x_1}
$$

### **6️⃣ Final Linear Regression Formula**  
After calculating **\(k\) and \(b\)**, our equation is:  

$$
y = kx + b
$$


---

### **🔥 Summary Table**
| Concept | Formula | Meaning |
|---------|---------|---------|
| **Variance (\(\text{Var}(X)\))** | $$ \frac{\sum (x_i - \bar{x})^2}{n} $$ | Measures spread of \(X\) |
| **Covariance (\(\text{Cov}(X, Y)\))** | $$ \frac{\sum (x_i - \bar{x}) (y_i - \bar{y})}{n} $$ | Measures relationship between \(X, Y\) |
| **Slope (\(k\))** | $$ \frac{\text{Cov}(X, Y)}{\text{Var}(X)} $$ | Rate of change of \(Y\) with respect to \(X\) |
| **Intercept (\(b\))** | $$ \bar{y} - k\bar{x} $$ | Where the line crosses \(Y\)-axis |
| **Linear Regression Equation** | $$ y = kx + b $$ | Best fit line for the data |

---

### **🎯 Final Takeaways**
✅ **Variance** shows how spread out \(X\) is.  
✅ **Covariance** shows how \(X\) and \(Y\) are related.  
✅ **Slope** is found by dividing covariance by variance.  
✅ **Intercept** is found using the mean values.  
✅ This method **uses all data points** for a more reliable fit compared to "rise over run" with just two points.  
