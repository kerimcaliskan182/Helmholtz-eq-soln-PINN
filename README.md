# Physics-Informed Neural Networks for PDEs

This repository contains an implementation of Physics-Informed Neural Networks (PINNs) specifically designed for solving the Helmholtz equation, a partial differential equation (PDE) with applications in astrophysics and plasma physics. The implementation is inspired by the concepts introduced in the paper "A hands-on introduction to Physics-Informed Neural Networks for solving partial differential equations with benchmark tests taken from astrophysics and plasma physics" by Hubert Baty.

## Introduction

PINNs leverage the power of deep learning to solve complex PDEs by incorporating the physics of the problem directly into the loss function. This focused approach on the Helmholtz equation allows for the exploration of solutions without the need for traditional mesh-based methods, offering significant advantages in terms of flexibility and computational efficiency. By focusing specifically on the Helmholtz equation, this project aims to demonstrate the versatility and power of PINNs in addressing the challenges associated with solving PDEs in the fields of astrophysics and plasma physics.


## Features

- Implementation of PINNs using PyTorch, a leading deep learning framework.
- Demonstration of PINNs' capability to solve PDEs with boundary conditions typical in astrophysics and plasma physics.
- Use of advanced sampling techniques for data preparation and model training.
- Visualization of the solution and comparison with analytical solutions where available.

## Installation

To run the code in this repository, you will need:

- Python 3.6 or later
- PyTorch 1.x
- NumPy
- Matplotlib
- SciPy

You can install the required libraries using pip:

```bash
pip install torch 
pip install numpy 
pip install matplotlib 
pip install scipy
```

## Usage

After cloning the repository, you can run the main script to train the PINN model and visualize the results.

## Acknowledgments

Special thanks to Hubert Baty for the insightful paper "A hands-on introduction to Physics-Informed Neural Networks for solving partial differential equations with benchmark tests taken from astrophysics and plasma physics", which inspired this implementation. The concepts and methodologies presented in the paper have been instrumental in guiding this work.

Additionally, this project makes use of PyTorch for the implementation of the neural network model. PyTorch's flexibility and ease of use have significantly contributed to the development and experimentation process.

## Contributing

Contributions to this project are welcome. Please feel free to submit issues or pull requests with improvements or new features.
