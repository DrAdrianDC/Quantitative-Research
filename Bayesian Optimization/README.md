## Bayesian Optimization
This project demonstrates how to use **Bayesian Optimization** (via `scikit-optimize`) to find the minimum of a simple two-variable function. The optimization is performed using **Gaussian Process Regression (GPR)** as a surrogate model, which efficiently balances exploration and exploitation to locate the global minimum with minimal function evaluations.



### Overview

Bayesian Optimization is a powerful method for optimizing expensive or unknown functions, especially when derivative information is unavailable. It is widely used in:


- **Hyperparameter tuning** for machine learning models  
- **Experimental design**  
- **Engineering optimization problems**

In this example, we minimize the function:

$$
f(x_1, x_2) = (x_1 - 2)^2 + (x_2 - 3)^2
$$


The true minimum is at **(x₁, x₂) = (2, 3)** with **f(x₁, x₂) = 0**, and the goal of Bayesian Optimization is to find this point efficiently.


