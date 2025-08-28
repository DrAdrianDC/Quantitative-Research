## Physics-Informed Neural Networks (PINNs) for Solving Differential Equations


 ![red text](https://img.shields.io/badge/WIP-red)


### What Are Physics-Informed Neural Networks (PINNs)?

**Physics-informed neural networks (PINNs)** are neural networks that incorporate physical laws described by differential equations into their loss functions to guide the learning process toward solutions that are more consistent with the underlying physics. PINNs can be used to:

   - Approximate solutions to partial differential equations (PDEs) and ordinary differential equations (ODEs).
   - Solve inverse problems, such as estimating model parameters from limited data.



### Project Motivation

PINNs leverage neural networks that incorporate physics laws into their loss function, ensuring solutions respect both data-driven patterns and governing equations. The project visualizes the comparison between the analytical solution and the model's solution using an animated GIF.




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
