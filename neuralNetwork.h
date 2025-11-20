#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include "matrix.h" // Matrix engine 

class NeuralNetwork {
private:
    // 1. Architecture Configurations
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    double learning_rate; // How fast it learns

    // 2. Memory (Matrices)
    Matrix weights_ih; // Weights from Input to Hidden
    Matrix weights_ho; // Weights from Hidden to Output

    // 3. The Bias
    Matrix bias_h; // Bias for Hidden Layer
    Matrix bias_o; // Bias for Output Layer

    // 4. Activation Function
    static double sigmoid(double x);

    // 5. Derivative of Activation Function
    static double dsigmoid(double y);

public:
    // Cosntructor : Initialize the brain size
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes);

    // Prediction Engine
    // Takes a standard C++ vector as input (list of numbers
    // Returns a standard C++ vector as output (list of probabilities)
    std::vector<double> feedForward(std::vector<double> input_array);

    // Training function
    // Input - data to look at
    // Target - answer it should have given
    void train(std::vector<double> input_array, std::vector<double> target_array);

};


#endif // NEURALNETWORK_H