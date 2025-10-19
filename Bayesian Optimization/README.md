## Bayesian Optimization
This project demonstrates how to use **Bayesian Optimization** (via `scikit-optimize`) to find the minimum of a simple two-variable function. The optimization is performed using **Gaussian Process Regression (GPR)** as a surrogate model, which efficiently balances exploration and exploitation to locate the global minimum with minimal function evaluations.



### Overview

Bayesian Optimization is a powerful method for optimizing expensive or unknown functions, especially when derivative information is unavailable. It is widely used in:
