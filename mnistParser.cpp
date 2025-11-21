#include "mnistParser.h"
#include <fstream>
#include <iostream>

// Helper function :
/*
    Goal : Read 4 bytes from file and flip them from high endian to
    little endian.

*/

int readInt(std::ifstream &file)
{
    unsigned char bytes[4];
    file.read((char *)bytes, 4); // Read raw 4 bytes
    /*
    Bitwise
    Shift byte 0 to 24 bits left
    Shift byte 1 to 16 bits left
    Shift byte 2 to 8 bits left
    leave byte 3 at end
    combine them using bitwise OR
    */
    return (bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3]);
}

namespace MNISTParser
{

    // Load images
    std::vector<std::vector<double>> loadImages(std::string filename)
    {
        std::vector<std::vector<double>> images;

        // Open file in binary mode;
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
        {
            std::cerr << "ERROR Cannot open file" << filename << std::endl;
            return images; // empty
        }

        // Step 1 : Read header : 16 bytes
        int magic_number = readInt(file);
        int number_of_images = readInt(file);
        int rows = readInt(file);
        int cols = readInt(file);

        /*
            Why same value to all integers :
            its not same, they will all be different
            bcz file variable (file) has memory its called cursor it moves
            Byte 0-3 : Consumed for magic_number. (Pointer moves to 4)
            Byte 4-7 : Consumed for number_of_images. (Pointer moves to 8)
            Byte 8-11 : Consumed for rows. (Pointer moves to 12)
            Byte 12-15 : Consumed for cols. (Pointer moves to 16)
            then byte 16 is the very first pixel of first image
        */

        // Validation : Magic number for images is always 2051
        if (magic_number != 2051)
        {
            std::cerr << "[ERROR] Invalid Image File! Magic Number: " << magic_number << std::endl;
            return images;
        }

        std::cout << "[PARSER] Loading " << number_of_images << " images..." << std::endl;
        // Step 2 : read pixels
        images.resize(number_of_images);
        int pixel_count = rows * cols; // 28*28=784

        for (int i = 0; i < number_of_images; i++)
        {
            images[i].resize(pixel_count);

            // read every pixel for this image
            for (int j = 0; j < pixel_count; j++)
            {
                unsigned char temp = 0;
                file.read((char *)&temp, 1); // read 1 byte

                // Normalize 0-255 -> 0.0-1.0
                images[i][j] = (double)temp / 255.0;
            }
        }
        std::cout << "[PARSER] Images Loaded Successfully." << std::endl;
        return images;
    }

    // Load labels
    std::vector<std::vector<double>> loadLabels(std::string filename)
    {
        std::vector<std::vector<double>> labels;

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
            return labels;
        }

        //  Step 1 : Read Header (8 Bytes)
        int magic_number = readInt(file);
        int number_of_labels = readInt(file);

        // Validation: Magic Number for Labels is always 2049
        if (magic_number != 2049)
        {
            std::cerr << "[ERROR] Invalid Label File! Magic Number: " << magic_number << std::endl;
            return labels;
        }

        std::cout << "[PARSER] Loading " << number_of_labels << " labels..." << std::endl;

        //  Step 2 : Read Answers
        labels.resize(number_of_labels);

        for (int i = 0; i < number_of_labels; i++)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, 1); // Read the digit (e.g., 5)

            // Convert "5" -> One-Hot Vector
            // [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            labels[i].resize(10, 0.0);
            labels[i][(int)temp] = 1.0;
        }

        std::cout << "[PARSER] Labels Loaded Successfully." << std::endl;
        return labels;
    }

}