#include <iostream>
#include <vector>
#include <ctime>    // For time seeding
#include "NeuralNetwork.h"

int main() {
    // 1. Seed the Randomizer
    // If we don't do this, we get the exact same random weights every time.
    std::srand(std::time(0));

    std::cout << "========== DAY 2 TEST: FEED FORWARD ==========" << std::endl;

    // 2. Instantiate the Brain
    // Architecture: 2 Input Nodes, 3 Hidden Nodes, 1 Output Node
    // This triggers the Constructor (allocating memory, randomizing weights).
    std::cout << "[Step 1] Initializing Neural Network (2-3-1)..." << std::endl;
    NeuralNetwork nn(2, 3, 1);

    // 3. Prepare Input Data
    // We are giving it a classic logic gate input: [1, 0]
    std::vector<double> input_data;
    input_data.push_back(1.0);
    input_data.push_back(0.0);

    std::cout << "[Step 2] Feeding input: [1.0, 0.0]" << std::endl;

    // 4. The Moment of Truth
    // This calls your feedForward function.
    // It converts Vector -> Matrix -> Multiplies -> Adds Bias -> Sigmoid -> Returns Vector
    std::vector<double> output = nn.feedForward(input_data);

    // 5. Display Result
    std::cout << "[Step 3] Brain Prediction: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 6. Interpretation
    std::cout << "----------------------------------------------" << std::endl;
    if (output.size() > 0 && output[0] >= 0.0 && output[0] <= 1.0) {
        std::cout << "SUCCESS: The Output is a valid probability (0-1)." << std::endl;
        std::cout << "The math pipeline is unbroken." << std::endl;
    } else {
        std::cout << "FAILURE: Output is invalid or empty." << std::endl;
    }
    std::cout << "==============================================" << std::endl;

    return 0;
}