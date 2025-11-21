#include "Matrix.h" // Links out header file
#include <cmath> // For math functions
#include <cstdlib> // For rand()
#include <ctime> // For seeding time
#include <iostream> // For printing

// 1. Constructor
Matrix::Matrix(int r, int c) {
    // TODO: Assign rows, cols, and resize data
    rows = r;
    cols = c;
    //memory allocation for the vector
    data.resize(rows * cols, 0.0); //row x col slots needed and all initialzed with zero
}

// 2. The Accessor
double& Matrix::at(int r, int c) {
    // TODO: Return data at index
    // inside computer there is no 2D its just 1D so we need to convert 2D to 1D
    // Formula : (Desired Row * Total Columns) + Desired Column
    // Like to access element at (2,3) in a 4 column matrix : (2*4)+3 = 11
    // Why double& ? because we want to return address of number not copy of number
    // This allows us to modify the number directly
    // Saves memory by not making a copy
    return data[(r * cols) + c];
}

const double& Matrix::at(int r, int c) const {
    return data[(r * cols) + c];
}

// 3. Print (So you can see what you built)
void Matrix::print() {
    // TODO: Double loop to print
    for (int i = 0; i < rows; i++){ // Walk across rows
        for (int j = 0; j < cols; j++){// Walk across columns
            std::cout << at(i,j) << "\t"; // \t is tab space
        }
        std::cout << std::endl; // New line after each row
    }
}

// 4. Randomize (Fill with random values)
/*  
    Why Randomize?
    If we initialize every weight with 0 or any same number, every neuron in the next layer 
    will perform same calculation and learn same features during training .
    This symmetry prevents the network from learning effectively.
    By randomizing weights, we break this symmetry, allowing different neurons to learn
    different features from the input data, which enhances the network's ability to learn complex patterns.
    When we calculate the error (Backpropagation), the math says "Neuron A and B made exact same mistake"
    So both will get same adjustment. This means they will continue to be identical.
    If same no matter how long you train, they will learn nothing. 

    Why between -1 and 1?
    We use -1 to 1 to keep the math in the "Active Zone" of the Sigmoid function. 
    If we stray too far, the math flatlines.
*/
void Matrix::randomize() {
    // Loop through every slot in the 1D vector
    // data.size() tells us how many slots we have
    for(int i = 0; i < data.size(); i++){
        // rand() gives a random integer like 1605
        // RAND_MAX is the biggest possible random number
        // Division gives us a decimal between 0 and 1
        double randomValue = (double) rand() / RAND_MAX;
        data[i] = randomValue * 2 - 1;
        // This is a math trick to get number in range of -1 to 1
        // If value is 0.0 -> (0 * 2) - 1 = -1
        // If value is 0.5 -> (0.5 * 2) - 1 = 0
        // If value is 1.0 -> (1 * 2) - 1 = 1
     }
}

// 5. Transpose (Flip rows and columns)
Matrix Matrix::transpose() {
    Matrix result(cols, rows); // Note the flipped dimensions
    for(int i = 0 ;i < rows; i++){
        for(int j = 0 ; j < cols; j++){
            // Put the value at (i,j) into (j,i)
            result.at(j, i) = at(i,j);
        }
    }
    return result;
}

// 6. Multiply by Scalar (Scale the matrix)
Matrix Matrix::multiplyScalar(double scalar) {
    Matrix result(rows, cols); // Same dimensions
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result.at(i,j) = at(i,j) * scalar;
        }
    }
    return result;
}

// 7. Add another Matrix
Matrix Matrix::add(const Matrix &m){
    if(rows != m.rows || cols != m.cols){
        std::cerr << "Error : Matrix dimensions Mismatch in addition. " << std::endl;
        return Matrix(0,0); // Return empty matrix on error
    }

    Matrix result(rows, cols); // Same dimensions
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result.at(i,j) = at(i,j) + m.at(i,j);
        }
    }
    return result;
}

// 8. Multiply by another Matrix
Matrix Matrix::multiply(const Matrix &m){
    if(cols != m.rows){
        std::cerr << "Error : Matrix dimensions Mismatch in multiplication. " << std::endl;
        return Matrix(0,0); // Return empty matrix on error
    }

    Matrix result(rows, m.cols); // New dimensions
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < m.cols; j++){
            double sum = 0.0;
            for(int k = 0; k < cols; k++){ // or k < m.rows
                sum += at(i,k) * m.at(k,j);
            }
            result.at(i,j) = sum;
        }
    }
    return result;
}

// 9. Map Function (Apply a function to every element)
// Inside Matrix.cpp

Matrix Matrix::map(double (*func)(double)) {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 1. Get the original value
            double val = at(i, j);
            
            // 2. Apply the function (Sigmoid) to it
            double activated = func(val);
            
            // 3. Store it in the result
            result.at(i, j) = activated;
        }
    }
    return result;
}

// Substraction function 
Matrix Matrix::subtract(const Matrix &m){
    if(rows != m.rows || cols != m.cols){
        std::cerr << "Error : Matrix dimensions Mismatch in subtraction. " << std::endl;
        return Matrix(0,0); // Return empty matrix on error
    }

    Matrix result(rows, cols); // Same dimensions
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result.at(i,j) = at(i,j) - m.at(i,j);
        }
    }
    return result;
}


// Hadamard Product (Element-wise multiplication)
Matrix Matrix::multiplyHadamard(const Matrix &m){
    
    if(rows != m.rows || cols != m.cols){
        std::cerr << "Error : Matrix dimensions Mismatch in Hadamard multiplication. " << std::endl;
        return Matrix(0,0); // Return empty matrix on error
    }

    Matrix result(rows, cols); // Same dimensions
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result.at(i,j) = at(i,j) * m.at(i,j);
        }
    }
    return result;
}
