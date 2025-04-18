# Stage 1.0

## TL;DR

### Info

- Linear Regression
- **Input** Features: **1**
- **Output** Features: **1**
- Loss: `MSE`
- Optimizer: None

### Limitations

- NO dedicated **Optimizer** 
- Manual weight update `w = w - gradient`
- NO batch support

## Implemntation

### _Content_:

1. Data Generation
2. Preprocessing the dataset
3. Pre-Training Setup
4. Training
5. Evaluation & Visualization
 
### _Requirements_:

1. numpy
2. matplotlib
3. scikit-learn

## Results:


<div style="width:100%; display:flex; flex-direction:column; align-items:center; gap:20px;">

<img src='./img/chart-1.png' style='width:80%' alt='Loss vs Epoch'>

<img src='./img/chart-2.png' style='width:80%' alt='Parameters and Gradients vs Epoch'>

<img src='./img/chart-3.png' style='width:80%' alt='Predictions on Test Data'>

</div>
