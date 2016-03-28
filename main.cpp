//
//  main.cpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//


#include "Rectangles.hpp"
#include "ImagePatch.hpp"
#include <iostream>


using namespace std;

int main(int argc, const char * argv[]) {
    
    int patchSize{25};
    int nbRectangle{100};
    int threshGradMagnitude{10};
    int nbRectPart{5};
    
    // call constructor of Rectangles to initialise the position of the rectangles as boundingBox(sR, sC, width, height)
    std::vector<std::array<int, 4>> bBox = Rectangles{patchSize, nbRectangle}.generate_rectangles();
    //std::vector<std::array<int, 4>> bBox = Rectangles().generate_rectangles();
    
    // call constructor of ImagePatch to initialise the threshold and the number of rectangles used for the orientation feature representation
    ImagePatch feature{threshGradMagnitude, nbRectPart};
    feature.extract_features_of_patches(bBox);
    
    
    mexFunctionTrain(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
    return 0;
}
