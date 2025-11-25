# Neural Network From Scratch (C++)

A fully connected feed-forward neural network (MLP) implemented entirely in C++. No TensorFlow, no PyTorch, no helper libraries. Just math, memory, and a stubborn need to see what happens under the hood.

Two major problems had to be solved manually:

**1. Data Handling**  
MNIST comes in a legacy `.idx` binary format. This project includes a custom parser that reads raw byte streams, converts Big-Endian to Little-Endian, manages memory safely, and reshapes data for training.

**2. Linear Algebra**  
With no NumPy or BLAS, a lightweight matrix engine was built from scratch to support multiplication, transposition, Hadamard products, and activation mapping.

---

## Features

### Matrix Engine (`matrix.cpp/h`)
- Matrix multiplication (O(n³))
- Transpose operations
- Hadamard (element-wise) products
- Scalar operations and activation mapping
- Efficient 1D storage with 2D indexing

### MNIST Binary Parser (`mnistParser.cpp/h`)
- Reads IDX file format
- Converts Big-Endian to Little-Endian
- Normalizes pixel values (0–1)
- One-hot encodes labels

### Neural Network Core (`neuralNetwork.cpp/h`)
- Feedforward propagation
- Backpropagation with gradient descent
- Sigmoid activation + derivative
- Configurable learning rate
- Random weight initialization

### Visualization
- ASCII digit rendering in terminal
- Real-time training progress
- Prediction confidence output

---


### License
Open source. Intended for learning, experimentation, and understanding the fundamentals of neural networks.

### Acknowledgments
Built as a ground-up exploration of neural network mathematics, data handling, and implementation details in pure C++.

