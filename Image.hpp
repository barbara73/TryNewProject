//
//  Image.hpp
//  TryNewProject
//
//  Created by Barbara Jesacher on 28.03.16.
//  Copyright (c) 2016 otl. All rights reserved.
//

#ifndef Image_hpp
#define Image_hpp

#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "xgboost/c_api.h"
#include <dmlc/timer.h>

class Image {
    int stride;
    int nbReductions;
    float reductionParameter;
    int blockWidth, blockHeight;
    int imgHeight;
    int imgWidth;
    BoosterHandle classifier;
    cv::Mat newMap = cv::Mat::zeros(imgHeight, imgWidth, CV_32F);
public:
    cv::Mat downscale_image(int, int, cv::Mat, int, int);
    std::vector<std::vector<float>> slide_window_over_image(int, cv::Mat, int, int);
    float testTheDataXGBoost(BoosterHandle, std::vector<float>, int, int);
        
    void set_imageHeight(const int h) {
        imgHeight = h;
    }
    
    void set_imageWidth(const int w) {
        imgWidth = w;
    }
    
    void set_hBooster(const BoosterHandle bh) {
        classifier = bh;
    }
    
    Image();
    Image(int, int, float);
    ~Image();
};

#endif /* Image_hpp */
