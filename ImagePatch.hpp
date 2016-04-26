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
#include "xgboost/c_api.h"
#include <dmlc/timer.h>



class ImagePatch {
    const int thresholdMagnitude;
    const int partialRectangleNB;
    unsigned long fileNameSize;
    int bLabel;
    //std::vector<std::array<int, 4>> bbRectangle;
    cv::Mat trainData, trainLabel;
public:
    static std::vector<float> make_orientationHistogramFeatures(cv::Mat&, std::vector<cv::Rect>&, int, int);
    int calculate_regionSum(cv::Mat, int, int, int, int, int);
    std::vector<std::vector<float>> extract_features_of_patches(std::vector<cv::Rect>&, cv::vector<cv::String>);
    std::vector<float> extract_label_of_patches(int);
    static std::vector<float> group_to_orientations(cv::Mat, cv::Mat, std::vector<cv::Rect>&, int);
    BoosterHandle trainTheDataXGBoost(std::vector<std::vector<float>>, std::vector<float>, int, int, int);
    
    int get_NbRectangles()const {
        return this->partialRectangleNB;
    }
    
    int get_ThMagnitude()const {
        return this->thresholdMagnitude;
    }
    
    unsigned long get_SizeFileName()const {
        return this->fileNameSize;
    }
      
    ImagePatch();
    ImagePatch(int, int);
    ~ImagePatch();
};
#endif /* ImagePatch_hpp */