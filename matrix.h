#ifndef MATRIX_H // Guard to prevent multiple inclusions
#define MATRIX_H // 

#include <vector>
#include <iostream>

class Matrix {
private:
    int rows, cols;
    std::vector<double> data;
public:
    Matrix(int r, int c);
    double& at(int r, int c);
    const double& at(int r, int c) const;
    // Utility functions
    void randomize();
    void print();
    Matrix transpose();
    Matrix multiplyScalar(double scalar);
    Matrix add(const Matrix &m);
    Matrix subtract(const Matrix& m);
    Matrix multiply(const Matrix &m);
    Matrix map(double (*func)(double));
    Matrix multiplyHadamard(const Matrix &m);
};

#endif // MATRIX_H