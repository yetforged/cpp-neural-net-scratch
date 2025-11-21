# Neural Network from Scratch (C++)

A fully connected feedforward neural network (Multi-Layer Perceptron) built entirely from scratch in C++. No TensorFlow, no PyTorch, no external matrix libraries—just raw mathematics, memory management, made to get a deep understanding of how neural networks actually work under the hood.

In high-level environments like Python, training a model on MNIST is often as trivial as import mnist. The library abstracts away file I/O, byte decoding, and array shaping.In standard C++, these luxuries dont exist. To build this engine, I had to solve two fundamental engineering problems manually:The Data Problem: The MNIST dataset uses a legacy .idx binary format. I designed a custom Binary Parser to read raw byte streams, handle dynamic memory allocation, and perform High-Endian to Little-Endian bitwise shifting to ensure data compatibility with modern CPU architectures.The Math Problem: Without numpy or BLAS, I implemented a linear algebra engine from scratch to handle $O(n^3)$ Matrix Multiplication, Transposition, and Hadamard Products.

## Features

### Custom Matrix Engine (`matrix.cpp/h`)

- Matrix multiplication (O(n³) implementation)
- Transpose operations
- Hadamard (element-wise) products
- Scalar operations
- Function mapping for activation functions
- Memory-efficient 1D vector storage with 2D indexing

### Binary Data Parser (`mnistParser.cpp/h`)

- Reads MNIST IDX binary file format
- Handles Big-Endian to Little-Endian byte conversion
- Normalizes pixel values (0-255 → 0.0-1.0)
- Converts labels to one-hot encoded vectors

### Deep Learning Core (`neuralNetwork.cpp/h`)

- Feedforward propagation
- Backpropagation with gradient descent
- Sigmoid activation function and its derivative
- Configurable learning rate
- Weight initialization with random values to break symmetry

### Visualization

- ASCII art renderer for digit visualization in terminal
- Real-time training progress display
- Prediction confidence output

## Architecture

### MNIST Digit Recognition

```
Input Layer:    784 nodes (28×28 pixels)
Hidden Layer:   128 nodes (Sigmoid activation)
Output Layer:   10 nodes (Digits 0-9, Sigmoid activation)
Learning Rate:  0.1
```

**Performance:** Achieves ~94.8% accuracy on the MNIST test set (10,000 unseen images) after just 1 epoch of training on 60,000 images.

### XOR Logic Gate

```
Input Layer:    2 nodes
Hidden Layer:   4 nodes (Sigmoid activation)
Output Layer:   1 node (Sigmoid activation)
Learning Rate:  0.1
Epochs:         50,000
```

**Performance:** Successfully learns the non-linear XOR function, demonstrating the network's ability to solve problems that single-layer perceptrons cannot.

## Results

### MNIST Digit Recognition Output

![MNIST Digit Recognition Results](Doc/digitrecog.png)

The network successfully recognizes handwritten digits with ~94.8% accuracy. The visualization shows:

- ASCII art representation of the input digit
- Real-time training progress
- Final accuracy metrics on 10,000 test images

### XOR Logic Gate Training

![XOR Logic Gate Results](Doc/xor_result.png)

The network successfully learns the XOR function, a classic non-linear problem that demonstrates the power of hidden layers. The output shows perfect classification after 50,000 training epochs.

## Getting Started

### Prerequisites

- C++ compiler with C++11 support (g++, MinGW, or MSVC)
- MNIST dataset files (see below)

### MNIST Dataset Setup

Dataset available at: http://yann.lecun.com/exdb/mnist/

### Compilation

**For MNIST Digit Recognition:**

```bash
g++ digitRecog.cpp mnistParser.cpp matrix.cpp neuralNetwork.cpp -o digitRecog
```

**For XOR Logic Gate:**

```bash
g++ xor.cpp matrix.cpp neuralNetwork.cpp -o xor
```

### Execution

**Run Digit Recognition:**

```bash
./digitRecog      # Linux/Mac
digitRecog.exe    # Windows
```

**Run XOR Demo:**

```bash
./xor             # Linux/Mac
xor.exe           # Windows
```

## Key Concepts Demonstrated

- **Backpropagation:** Error propagation through network layers
- **Gradient Descent:** Iterative weight optimization
- **Activation Functions:** Non-linear transformations (Sigmoid)
- **Matrix Calculus:** Derivatives and chain rule application
- **One-Hot Encoding:** Label representation for classification
- **Normalization:** Feature scaling for better convergence
- **Binary File I/O:** Low-level data parsing
- **Endianness Conversion:** Cross-platform data compatibility

## License

This project is open source and available for educational purposes.

## Acknowledgments

Built as a deep dive into the mathematics and implementation details of neural networks.
