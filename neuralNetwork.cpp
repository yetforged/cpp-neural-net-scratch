#include "neuralNetwork.h"
#include "matrix.h"
#include <vector>
#include <cmath> // For exp function

// The constructor 
// Goal to set up topology and resize all matrices

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes)
    :input_nodes(input_nodes),
    hidden_nodes(hidden_nodes),
    output_nodes(output_nodes),
    // Initialize matrices with specific dimensions
    weights_ih(hidden_nodes, input_nodes),
    weights_ho(output_nodes, hidden_nodes),
    bias_h(hidden_nodes, 1),
    bias_o(output_nodes, 1)
// The above syntax has a weird colon after the constructor name
// It is called an initializer list
// It initializes member variables before the constructor body runs
// The initializer list constructs the object before entering the constructor body
// Why this needed?
// Because some member variables (like matrices) need to be initialized with parameters
// like number of rows and columns
// if we tried to put weights_ih = Matrix(h, i); inside the constructor body
// it would call the default constructor but would not know what size to make the matrix
// So we use the initializer list to pass the sizes directly when constructing the matrices
// This is more efficient and necessary for certain types of member variables
    {
        // Matrices above are full of zeros 
        // We need to randomize them to break symmetry
        weights_ih.randomize();
        weights_ho.randomize();
        bias_h.randomize();
        bias_o.randomize();

        learning_rate = 0.1; // Default learning rate
    }

/*  
    Weights inputs to hidden (weights_ih)
    size : hidden_nodes x input_nodes
    Weights hidden to output (weights_ho)
    size : output_nodes x hidden_nodes
    Bias hidden (bias_h)
    size : hidden_nodes x 1
    Bias output (bias_o)
    size : output_nodes x 1
    We want the hidden layer to have one column per input
    We want the output layer to have one column per hidden node
    We want the result to match hidden layer size 
*/

// Activation Function
// Returns a value between 0 and 1
// 1 / (1 + e^(-x))
/*
    No matter how big the number gets (e.g., 1,000,000), Sigmoid squishes it to 0.999. 
    No matter how negative it gets (e.g., -1,000,000), Sigmoid squishes it to 0.001.
*/
double NeuralNetwork::sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// The dsigmoid function - derivative of sigmoid
double NeuralNetwork::dsigmoid(double y){
    // y = sigmoid(x)
    return y * (1 - y);
}



// Feedforward function
/*
    1. Convert the normal C++ vector to a Matrix
    2. Layer 1 : Input -> Hidden
        a. Calculate WeightedSum = Weights_ih * inputs + bias_h
        b. Apply Activation Function to WeightedSum to get hidden layer outputs
    3. Layer 2 : Hidden -> Output
        a. Calculate WeightedSum = Weights_ho * hidden + bias_o
        b. Apply Activation Function to WeightedSum to get final outputs
    4. Convert output Matrix back to C++ vector and return it
*/

std::vector<double> NeuralNetwork::feedForward(std::vector<double> input_array){
    // 1. Vector to Matrix
    // We need tu turn list (eg [0.5, 0.2, 0.1]) into a column matrix
    // So our math engine can process it
    if (input_array.size() != input_nodes){
        std::cerr << "Error: Input size does not match number of input nodes." << std::endl;
        return std::vector<double>(); // Return empty vector on error
    }

    // Create matrix from the vector data manually
    Matrix inputs(input_nodes, 1);
    for(int i = 0 ; i < input_nodes; i++){
        inputs.at(i, 0) = input_array[i];
    }

    // 2. HIDDEN LAYER
    Matrix hidden = weights_ih.multiply(inputs); // Weighted sum
    hidden = hidden.add(bias_h); // Add bias

    // 3. Apply activation function to hidden layer
    hidden = hidden.map(sigmoid); // map function applies sigmoid to each element

    // 4. OUTPUT LAYER
    Matrix outputs = weights_ho.multiply(hidden); // Weighted sum
    outputs = outputs.add(bias_o); // Add bias
    outputs = outputs.map(sigmoid); // Apply activation function

    // 5. Matrix to Vector
    std::vector<double> result;
    for(int i=0; i<output_nodes;i++){
        result.push_back(outputs.at(i,0));
    }
    return result;
}

void NeuralNetwork::train(std::vector<double> input_array, std::vector<double> target_array) {
    
    // PHASE 1: FEED FORWARD :  AI Takes a Guess
    // Goal: Pass data from Input -> Hidden -> Output to get the current prediction.  
    //   Convert Inputs to Matrix
    if (input_array.size() != input_nodes || target_array.size() != output_nodes) {
        std::cerr << "Input or Target size mismatch!" << std::endl;
        return;
    }
    // Preparing the input matrix
    // Convert Inputs to Matrix  
    Matrix inputs(input_nodes, 1);
    for (int i = 0; i < input_nodes; i++) {
        inputs.at(i, 0) = input_array[i];
    }
    
    //   Calculate Hidden Layer Output
    // Inputs -> Hidden
    // Math : hidden = sigmoid(weights_ih * inputs + bias_h)
    Matrix hidden = weights_ih.multiply(inputs); // Weighted sum // dot product
    hidden = hidden.add(bias_h); // Add bias
    hidden = hidden.map(sigmoid); // Activation : Squishes to 0-1
   
    
    //   Calculate Final Output
    // Hidden -> Output
    // Math : outputs = sigmoid(weights_ho * hidden + bias_o)
    Matrix outputs = weights_ho.multiply(hidden); // Weighted sum
    outputs = outputs.add(bias_o); // Add bias
    outputs = outputs.map(sigmoid); // Activation
    
    // PHASE 2: BACKPROPAGATION (Who responsible for the error?)
    // Goal: Calculate errors and check how much each weight contributed to the error.
    
    // target vector to target Matrix
    Matrix target(output_nodes, 1);
    for(int i = 0; i < output_nodes; i++) {
        target.at(i, 0) = target_array[i];
    }
    //   Calculate Output Error
    // ERROR = TARGETS - OUTPUTS
    // Example: Wanted 1.0, got 0.2. Error = 0.8 (We need to go UP).
    Matrix output_errors = target.subtract(outputs);

    //   Calculate Hidden Error
    // ERROR_HIDDEN = WEIGHTS_HO_TRANSPOSED * ERROR_OUTPUT
    // We dont have a target for hidden layer
    // We calculate its error by seeing how much it contributed to the output error
    // We send the error back through the weights
    // We need to transpose weights_ho to match dimensions
    // Why transpose?
    // Forward : Hidden(2 x 1) -> Weights_ho(1 x 2) -> Output(1 x 1)
    // Backward : Output_Error(1 x 1) -> Hidden_Error(2 x 1) needs Weights(2 x 1)
    Matrix weights_ho_T = weights_ho.transpose();
    Matrix hidden_errors = weights_ho_T.multiply(output_errors);

    // PHASE 3: GRADIENT DESCENT (Update the Weights)
    // Goal : Nudge the waits to reduce error next time.
    // Formula : New Weight = Old Weight + (Learning Rate * Gradient * Input)
   
    
    //  Calculate Gradients (Nudges)
    // Gradient = Error * dsigmoid(Output) * LearningRate
    // Logic:
    // if output was close to 0 or 1, dsigmoid is small -> small change(dont change much)
    // if output was around 0.5, dsigmoid is large -> large change (change more)
     Matrix gradients = outputs.map(dsigmoid); // Derivative of outputs (calculating slope)
    gradients = gradients.multiplyHadamard(output_errors); // Element-wise multiplication
    gradients = gradients.multiplyScalar(learning_rate); 
    // Scale by learning rate
    // Big Error = Big Change. Small Error = Small Change.
    // Note: We use Hadamard (Element-wise) because each neuron has its own error.
    //   Adjust Weights (Hidden -> Output)
    // Delta = Gradient * Hidden_Transposed
    // Weights_HO = Weights_HO + Delta
    Matrix hidden_T = hidden.transpose();
    Matrix weight_ho_deltas = gradients.multiply(hidden_T);
    weights_ho = weights_ho.add(weight_ho_deltas);
    bias_o = bias_o.add(gradients); // Adjust the output bias
    //   Adjust Weights (Input -> Hidden)
    // Delta = Hidden_Gradient * Input_Transposed
    // Weights_IH = Weights_IH + Delta
    // Calculate Hidden Gradient
    Matrix hidden_gradients = hidden.map(dsigmoid);
    hidden_gradients = hidden_gradients.multiplyHadamard(hidden_errors);
    hidden_gradients = hidden_gradients.multiplyScalar(learning_rate);

    // Calculate deltas for input to hidden weights
    Matrix inputs_T = inputs.transpose();
    Matrix weight_ih_deltas = hidden_gradients.multiply(inputs_T);
    weights_ih = weights_ih.add(weight_ih_deltas); // Update input to hidden weights
    bias_h = bias_h.add(hidden_gradients); // Adjust the hidden bias

}