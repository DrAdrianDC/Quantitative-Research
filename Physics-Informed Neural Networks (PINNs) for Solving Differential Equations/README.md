## Physics-Informed Neural Networks (PINNs) for Solving Differential Equations

### Importance and Potential Applications in Quantitative Analytics

Physics-Informed Neural Networks (PINNs) are an innovative approach that integrates physical laws with machine learning models, providing an efficient way to solve complex partial differential equations. This is particularly useful in quantitative analytics, where the low of data availability is an issue in some biological and engineering systems for instance, which can make other Machine Learning techniques  ineffective. The PINNs also do not requiere labeled data, like results from experiments or simulations.

### Project Motivation

This project demonstrates the application of Physics-Informed Neural Networks (PINNs) to solve partial differential equations. PINNs leverage neural networks that incorporate physics laws into their loss function, ensuring solutions respect both data-driven patterns and governing equations. The project visualizes the comparison between the analytical solution and the model's solution using an animated GIF.


### Overview

* Trains a PINN model to approximate the solution of a given differential equation.

* Visualizes both the analytical and model-predicted solutions.

* Generates an animated GIF to demonstrate the comparison over time or spatial coordinates.


### Usage

1- Ensure that the differential equation parameters are correctly defined in the Jupyter Notebook.

2- Run the notebook to train the model and generate the animated GIF:

```bash
jupyter notebook PINN_solution.ipynb
```

3- The resulting GIF is saved in the results/ directory.
