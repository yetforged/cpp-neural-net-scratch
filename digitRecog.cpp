#include <iostream>
#include <vector>
#include <algorithm> // For std::max_element
#include <iomanip>   // For nice output formatting
#include "NeuralNetwork.h"
#include "MnistParser.h"

// CONSTANTS (File Paths)

// Make sure your 'data' folder is in the same directory as your executable!
const std::string TRAIN_IMAGES = "data/train-images-idx3-ubyte/train-images.idx3-ubyte";
const std::string TRAIN_LABELS = "data/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
const std::string TEST_IMAGES = "data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
const std::string TEST_LABELS = "data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";

// VISUALIZATION HELPER
/*
   Goal: Print the 28x28 pixel grid to the terminal.
   - If pixel > 0.5 (White), print "@"
   - If pixel < 0.5 (Black), print "."
   This verifies that our data isn't corrupt.
*/
void printDigit(const std::vector<double> &pixels, int label)
{
    std::cout << "\n--- DIGIT VISUALIZER (Label: " << label << ") ---" << std::endl;

    for (int i = 0; i < 28; i++)
    { // Rows
        for (int j = 0; j < 28; j++)
        { // Cols
            // Calculate Index in the flat 1D vector
            int index = i * 28 + j;

            // Visual Threshold
            if (pixels[index] > 0.5)
                std::cout << " @";
            else
                std::cout << " .";
        }
        std::cout << std::endl; // New line after every row
    }
    // std::cout << "----------------------------------------\n"
    //  << std::endl;
}

// ARGMAX HELPER
/*
   Goal: Find the index of the highest probability.
   - Input: [0.1, 0.0, 0.8, 0.1]
   - Output: 2 (Because 0.8 is the biggest number)
   This turns the AI's probability vector into a single predicted digit.
*/
int getPrediction(const std::vector<double> &output)
{
    // std::max_element returns an iterator to the max value.
    // std::distance calculates the index.
    auto max_iter = std::max_element(output.begin(), output.end());
    return std::distance(output.begin(), max_iter);
}

int main()
{
    std::cout << "DIGIT RECOGNIZER" << std::endl;

    //  STEP 1 : LOAD DATA
    std::cout << "\nSTEP 1 Loading MNIST Data..." << std::endl;

    // Load Training Data
    std::vector<std::vector<double>> train_images = MNISTParser::loadImages(TRAIN_IMAGES);
    std::vector<std::vector<double>> train_labels = MNISTParser::loadLabels(TRAIN_LABELS);

    // Load Test Data
    std::vector<std::vector<double>> test_images = MNISTParser::loadImages(TEST_IMAGES);
    std::vector<std::vector<double>> test_labels = MNISTParser::loadLabels(TEST_LABELS);

    // Safety Check
    if (train_images.empty() || train_labels.empty())
    {
        std::cerr << " Could not load data. Exiting." << std::endl;
        return 1;
    }

    //  STEP 2 : INITIALIZE BRAIN
    std::cout << "\nSTEP 2 Initializing Neural Network..." << std::endl;
    // Input : 784 (28x28 pixels)
    // Hidden : 128 (Enough capacity to learn shapes)
    // Output : 10 (Digits 0-9)
    NeuralNetwork nn(784, 128, 10);
    std::cout << "Topology: 784 -> 128 -> 10" << std::endl;

    //  STEP 3 : TRAINING
    std::cout << "\nSTEP 3 Training ..." << std::endl;

    // We train on the full 60,000 dataset once (1 Epoch)
    // Or multiple times if we want higher accuracy.
    int dataset_size = train_images.size();
    int epochs = 1;

    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < dataset_size; i++)
        {
            // Train on one image
            nn.train(train_images[i], train_labels[i]);

            // Progress Log (Every 100 images)
            if (i % 100 == 0)
            {
                // Calculate current accuracy on this specific example
                std::vector<double> out = nn.feedForward(train_images[i]);
                int guess = getPrediction(out);
                int actual = getPrediction(train_labels[i]); // Find which index is 1.0

                std::cout << "Epoch " << e + 1 << " | Image " << i << " / " << dataset_size
                          << " | Guess: " << guess << " (Target: " << actual << ")"
                          << " \r" << std::flush; // \r overwrites the line
            }
        }
    }
    std::cout << "\n\nSUCCESS :: Training Complete." << std::endl;

    std::cout << "\n TESTING..." << std::endl;
    //  STEP 4 : TESTING (ACCURACY)
    std::cout << "\nSTEP 4 Evaluating on Test Set (10,000 images)..." << std::endl;

    int correct = 0;
    int total_test = test_images.size();

    for (int i = 0; i < total_test; i++)
    {
        std::vector<double> output = nn.feedForward(test_images[i]);

        int guess = getPrediction(output);
        int actual = getPrediction(test_labels[i]);

        if (guess == actual)
        {
            correct++;
        }

        // Visual Check : Show the first 3 test cases
        if (i < 3)
        {
            printDigit(test_images[i], actual);
            std::cout << "AI Prediction: " << guess << "\n"
                      << std::endl;
        }
    }

    double accuracy = ((double)correct / total_test) * 100.0;
    std::cout << " FINAL ACCURACY: " << accuracy << "%" << std::endl;
    std::cout << " Correct: " << correct << " / " << total_test << std::endl;

    return 0;
}