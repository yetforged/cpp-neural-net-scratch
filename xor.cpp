#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iomanip> // For std::setw, std::setprecision

#include "NeuralNetwork.h"

// Helper to print a progress bar
void printProgressBar(int current, int total) {
    float progress = (float)current / total;
    int barWidth = 50;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main() {
    std::srand(std::time(0));

    
    std::cout << "   NEURAL NETWORK: NON-LINEAR LOGIC GATE (XOR)   " << std::endl;
    

    // 1. Initialize Brain
    // 2 Inputs -> 4 Hidden -> 1 Output
    NeuralNetwork nn(2, 4, 1);
    std::cout << "[SYSTEM] Architecture: 2-4-1 Perceptron" << std::endl;
    std::cout << "[SYSTEM] Learning Rate: 0.1" << std::endl;
    std::cout << "[SYSTEM] Activation: Sigmoid" << std::endl;

    // 2. Training Data
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<std::vector<double>> targets = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };

    // 3. Training Loop
    int epochs = 50000;
    std::cout << "\n[PROCESS] Training Model (" << epochs << " epochs)..." << std::endl;

    for (int i = 0; i < epochs; i++) {
        int index = std::rand() % 4;
        nn.train(inputs[index], targets[index]);

        // Update progress bar every 500 iterations
        if (i % 500 == 0) {
            printProgressBar(i, epochs);
        }
    }
    printProgressBar(epochs, epochs); // Finish the bar
    std::cout << "\n\n[SUCCESS] Model Trained.\n" << std::endl;

    // 4. Evaluation Table
   
    std::cout << " INPUT A | INPUT B | TARGET | PREDICTION | STATUS " << std::endl;
    

    // Formatting setup for the table
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < 4; i++) {
        std::vector<double> output = nn.feedForward(inputs[i]);
        double guess = output[0];
        double target = targets[i][0];

        // Determine Pass/Fail based on rounding
        // If guess > 0.5, we treat it as 1. Else 0.
        double rounded_guess = (guess > 0.5) ? 1.0 : 0.0;
        bool pass = (rounded_guess == target);

        // Print Row
        std::cout << "    " << (int)inputs[i][0] << "    |    " 
                  << (int)inputs[i][1] << "    |   " 
                  << (int)target << "    |   " 
                  << guess << "   |  " 
                  << (pass ? "PASS" : "FAIL") << std::endl;
    }
    std::cout << "===================================================" << std::endl;

    return 0;
}