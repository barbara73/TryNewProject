//
//  ImagePatch.hpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//

#ifndef ImagePatch_hpp
#define ImagePatch_hpp

#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class ImagePatch {
    const int thresholdMagnitude;
    const int partialRectangleNB;
    std::vector<std::array<int, 4>> bbRectangle;
    cv::Mat image;
public:
    std::vector<std::vector<double>> make_orientationHistogramFeatures();
    int calculate_regionSum(cv::Mat, int, int, int, int, int);
    void extract_features_of_patches(std::vector<std::array<int, 4>>);
    
    ImagePatch();
    ImagePatch(int, int);
    ~ImagePatch();
};
#endif /* ImagePatch_hpp */
