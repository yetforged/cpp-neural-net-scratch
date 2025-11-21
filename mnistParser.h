#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

#include <vector>
#include <string>

/*
    The Problem : Parsing MNIST dataset files
    We have a file full of raw bytes (0s and 1s) that represent images and labels.
    We need to read these bytes and convert them into a usable format (like vectors or arrays)
    so that we can use them for training machine learning models.

    C++ doesnt have a built in function to read these files manually
    so we have to write our own parser.

    Challenges:
    1. Understanding the MNIST file format (headers, data structure)
    2. Reading binary files in C++
    3. Converting raw byte data into usable image and label formats
    4. Handling potential errors (file not found, incorrect format)
    5. Ensuring efficient memory usage when dealing with large datasets

    ENDIANNESS : 
    The MNIST file was created on high performance servers that store numbers in high-endian format aka big-endian.
    Most modern computers use little-endian format. (intel/amd processors)

    The "Reading Direction" (Endianness): Imagine writing the number Five Thousand (5000).
    Humans (and MNIST Files): Write it Left-to-Right: 5 0 0 0. (This is called Big Endian).
    Our CPU (Intel/AMD): Internally reads numbers Right-to-Left: 0 0 0 5. (This is called Little Endian).
    If we read the file directly, your computer will read 5000 as 5 (or garbage). 
    Our Parser must physically flip the bytes whenever it reads an Integer 
    

    Structure of MNIST Files:
    These files are like long, continuos streams of bytes.
    No spaces, no commas, no new lines. Just raw data.

    FILE 1: TRAINING IMAGES (train-images-idx3-ubyte)
    This file has 2 parts: "Header" (Info) and "Body" (pixels)
    
    Part A: Header (16 bytes)
    This tells us whats inside the file.
    We must flip these butes (High to Low) when reading them.

    [Bytes 0-3]  : Magic Number (2051) - Identifies the file type
    [Bytes 4-7]  : Number of Images (e.g., 60000)
    [Bytes 8-11] : Number of Rows (28 for MNIST)
    [Bytes 12-15]: Number of Columns (28 for MNIST)

    Part B: Body (Image Data)
    After the header, it is just a stream of pixel values (0-255).
    - Value 0: Black background
    - Value 255: White foreground

    Logic: 
    - Read the next 784 bytes (28*28). This is image 1
    - Read the next 784 bytes. This is image 2
    - Repeat this until we read all images. (60000 for this training set)

    FILE 2: TRAINING LABELS (train-labels-idx1-ubyte)
    This file also has 2 parts: "Header" (Info) and "Body"
    Part A: Header (8 bytes)
    [Bytes 0-3] : Magic Number (2049) - Identifies the file type (label file)
    [Bytes 4-7] : Number of Labels (e.g., 60000)    

    Part B: Body (Label Data)
    After the header, it is just a stream of label values (0-9).
    Each byte corresponds to the label of the image at the same index in the image file.
    -Byte #8: Label for Image 1 (eg. 5)
    -Byte #9: Label for Image 2 (eg. 0)
    - And so on...

    -------------------------------------------------------
    The header stores numbers in high endian format. (left to right)
    Our cpu reads numbers in low endian format. (right to left)
    So we must physically flip the bytes when reading integers from the header.

    eg the number is 2051
    Hex: 00 00 08 03 (big-endian)
    Our cpu reads it as: 03 08 00 00 (little-endian)
    This is 197120 in decimal, which is incorrect.
    We need to flip the bytes to get the correct number.

    We only flip the headers. Pixel data (single bytes) do not need flipping.


    */

    /*
        Why namespaces?
        Namespace is simply a labeled container used to group related functions together
        (e.g., MNISTParser::loadImages) to prevent their names from clashing with other
        parts of your code. We chose a namespace instead of a Class because our parser 
        is just a collection of "stateless tools"—it doesn't need to remember variables
        or manage memory between calls—so creating an object (e.g., Parser p = new Parser()) 
        would be unnecessary code clutter compared to just using the tool directly.
    */

namespace MNISTParser {

   // Load images 
   // input: path to the MNIST image file
   // Process : reads raw bytes, flips header endianness,  normalizes pixels (0-255 -> 0.0-1.0)
   // Output : A vector of images (each image is vector of 784 doubles)

   std::vector<std::vector<double>> loadImages(std::string filename);

    // Load labels
    // input: path to the MNIST label file
    // Process : reads raw bytes, flips header endianness
    //Output :  Vector of labels (each label is an integer 0-9)
    // eg label 5 will be [0,0,0,0,0,1,0,0,0,0] (one-hot encoded)

    std::vector<std::vector<double>> loadLabels(std::string filename);


} // namespace MNISTParser

#endif // MNIST_PARSER_H