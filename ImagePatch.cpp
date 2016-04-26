//
//  ImagePatch.cpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "ImagePatch.hpp"
//#include "Rectangles.hpp"
#include <iostream>
#include <iterator>
#include <chrono>

using namespace cv;


// constructors
//-------------
ImagePatch::ImagePatch():thresholdMagnitude{3}, partialRectangleNB{1111} {}

ImagePatch::ImagePatch(int th, int nbRect):thresholdMagnitude{th}, partialRectangleNB{nbRect} {}


// to view the matrix
//-------------------
template <class T>
void print(T & t, size_t rows, size_t columns)
{
    for(size_t i = 0;i < rows; ++i)
    {
        for(size_t j = 0;j < columns; ++j)
            printf("%g ", t[i][j]);
        
        printf("\n");
    }
    printf("\n");
}


// to view the vector
//-------------------
template <class T>
void printVector(T & t, size_t rows)
{
    for(size_t i = 0;i < rows; ++i)
    {
        printf("%d ", t[i]);
    }
    printf("\n");
}


// extract the features from the image patches
//--------------------------------------------
std::vector<std::vector<float>> ImagePatch::extract_features_of_patches(std::vector<cv::Rect>& bb, vector<String> fileName) {
    //bbRectangle = bb;
    fileNameSize = fileName.size();
    std::vector<float> orientationFeatures;
    std::vector<std::vector<float>> totalFeaturePerPatch;
    
    
    //for(size_t i = 0; i < fileNameSize; ++i)
    for(size_t i = 0; i < 1000; ++i)
    {
        Mat img = imread(fileName[i]);
               
        if(!img.data)
            std::cerr << "Problem loading image!!!" << std::endl;
        
        orientationFeatures = make_orientationHistogramFeatures(img, bb, thresholdMagnitude, partialRectangleNB);
        totalFeaturePerPatch.push_back(orientationFeatures);
        
    }
    //print(totalFeaturePerPatch, 10, partialRectangleNB*9);
    
    return totalFeaturePerPatch;
}


// make labels per patch
//----------------------
std::vector<float> ImagePatch::extract_label_of_patches(int label) {
    bLabel = label;
    //std::vector<int> labelPerPatch(static_cast<int>(fileNameSize), bLabel);
    std::vector<float> labelPerPatch(static_cast<int>(1000), bLabel);
    
    //printVector(labelPerPatch, 10);
    
    return labelPerPatch;
}


// calculate the orientation histogram features
//---------------------------------------------
std::vector<float> ImagePatch::make_orientationHistogramFeatures(Mat& image, std::vector<cv::Rect>& bb, int th, int nbRect) {
    
    Mat magnitude, direction, grad_y, grad_x, image_grey;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    
    // change colour image to greyscale image
    cvtColor(image, image_grey, CV_BGR2GRAY);
    
    // apply sobel to get gradient image
    Sobel(image_grey, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    Sobel(image_grey, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    
    bool angleInDegrees = true;
    
    // determine the gradient magnitude image and the gradient direction image
    cartToPolar(grad_x, grad_y, magnitude, direction, angleInDegrees);
    
    int const max_BINARY_value = 1;
    
    Mat orient0;                    //mask
    //int th = get_ThMagnitude();
    threshold(magnitude, orient0, th, max_BINARY_value, THRESH_BINARY);
    
    //std::vector<double> regionSum = group_to_orientations(direction, orient0);
    return group_to_orientations(direction, orient0, bb, nbRect);//regionSum;
}


// group gradients into 8 orientations
//------------------------------------
std::vector<float> ImagePatch::group_to_orientations(Mat dir, Mat edge0, std::vector<cv::Rect>& bbRectangle, int nbRect) {
   
    Mat orient1 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient2 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient3 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient4 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient5 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient6 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient7 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    Mat orient8 = Mat::zeros(dir.rows, dir.cols, CV_32F);
    
    for(int i=0; i<dir.rows; i++) {
        
        float* d = dir.ptr<float>(i);
        float* row1 = orient1.ptr<float>(i);
        float* row2 = orient2.ptr<float>(i);
        float* row3 = orient3.ptr<float>(i);
        float* row4 = orient4.ptr<float>(i);
        float* row5 = orient5.ptr<float>(i);
        float* row6 = orient6.ptr<float>(i);
        float* row7 = orient7.ptr<float>(i);
        float* row8 = orient8.ptr<float>(i);
        
        for(int j=0; j<dir.cols; j++) {
            
            float value = d[j];
            
            if (value >= 0 && value < 45)
                row1[j] = 1;
            else if (value >= 45 && value < 90)
                row2[j] = 1;
            else if (value >= 90 && value < 135)
                row3[j] = 1;
            else if (value >= 135 && value < 180)
                row4[j] = 1;
            else if (value >= 180 && value < 225)
                row5[j] = 1;
            else if (value >= 225 && value < 270)
                row6[j] = 1;
            else if (value >= 270 && value < 315)
                row7[j] = 1;
            else if (value >= 315 && value < 360)
                row8[j] = 1;
            else
                std::cout << "there is a mistake! degrees from [0, 360)" << std::endl;
        }
    }
    
    Mat integralImg0, integralImg1, integralImg2, integralImg3, integralImg4, integralImg5, integralImg6, integralImg7, integralImg8;
    Mat edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8;
    
    integral(edge0, integralImg0);
    multiply(edge0, orient1, edge1, 1, -1);
    integral(edge1, integralImg1);
    multiply(edge0, orient2, edge2, 1, -1);
    integral(edge2, integralImg2);
    multiply(edge0, orient3, edge3, 1, -1);
    integral(edge3, integralImg3);
    multiply(edge0, orient4, edge4, 1, -1);
    integral(edge4, integralImg4);
    multiply(edge0, orient5, edge5, 1, -1);
    integral(edge5, integralImg5);
    multiply(edge0, orient6, edge6, 1, -1);
    integral(edge6, integralImg6);
    multiply(edge0, orient7, edge7, 1, -1);
    integral(edge7, integralImg7);
    multiply(edge0, orient8, edge8, 1, -1);
    integral(edge8, integralImg8);
    
    /*// for test
     Mat o1, o2, o3, o4, o5, o6, o7, o8;
     multiply(orient0, orient1, o1, 1, -1);
     multiply(orient0, orient2, o2, 1, -1);
     multiply(orient0, orient3, o3, 1, -1);
     multiply(orient0, orient4, o4, 1, -1);
     multiply(orient0, orient5, o5, 1, -1);
     multiply(orient0, orient6, o6, 1, -1);
     multiply(orient0, orient7, o7, 1, -1);
     multiply(orient0, orient8, o8, 1, -1);
     
     Mat dst, dst1, dst2, dst3, dst4, dst5, dst6, comparison;
     add(o1, o2, dst, noArray(), -1);
     add(o3, dst, dst1, noArray(), -1);
     add(o4, dst1, dst2, noArray(), -1);
     add(o5, dst2, dst3, noArray(), -1);
     add(o6, dst3, dst4, noArray(), -1);
     add(o7, dst4, dst5, noArray(), -1);
     add(o8, dst5, dst6, noArray(), -1);
     
     compare(dst6, orient0, comparison, CMP_EQ);
     
     namedWindow("mask", CV_WINDOW_AUTOSIZE);
     imshow("mask", comparison);
     cv::waitKey(0);
     cv::destroyWindow("mask");*/
    
    // evaluate the sum of the patch for normalisation
    float max0, max1, max2, max3, max4, max5, max6, max7, max8;
    int x = integralImg0.cols-1;
    int y = integralImg0.rows-1;
    max0 = integralImg0.at<double>(Point(x,y));
    max1 = integralImg1.at<double>(Point(x,y));
    max2 = integralImg2.at<double>(Point(x,y));
    max3 = integralImg3.at<double>(Point(x,y));
    max4 = integralImg4.at<double>(Point(x,y));
    max5 = integralImg5.at<double>(Point(x,y));
    max6 = integralImg6.at<double>(Point(x,y));
    max7 = integralImg7.at<double>(Point(x,y));
    max8 = integralImg8.at<double>(Point(x,y));
    
    // calculate the region sum in a rectangle of each map
    std::vector<float> newSum {};
    
    //int partRectNb = get_NbRectangles();
    
    for (int i = 0; i != nbRect; ++i) {
        
        Rect bb = bbRectangle[i];
        //std::cout << bb << std::endl;
        /*int x = bb[0];
        int y = bb[i][1];
        int w = bbRectangle[i][2];
        int h = bbRectangle[i][3];*/
        
        //std::cout << x << ", " << y << ", " << w << ", " << h << ", " <<  std::endl;
        float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;
        /*
        sum0 = (integralImg0.at<double>(Point(x+w,y+h)) - integralImg0.at<double>(Point(x+w,y)) - integralImg0.at<double>(Point(x,y+h)) + integralImg0.at<double>(Point(x,y)))/(max0+0.000001);
        sum1 = (integralImg1.at<double>(Point(x+w,y+h)) - integralImg1.at<double>(Point(x+w,y)) - integralImg1.at<double>(Point(x,y+h)) + integralImg1.at<double>(Point(x,y)))/(max1+0.000001);
        sum2 = (integralImg2.at<double>(Point(x+w,y+h)) - integralImg2.at<double>(Point(x+w,y)) - integralImg2.at<double>(Point(x,y+h)) + integralImg2.at<double>(Point(x,y)))/(max2+0.000001);
        sum3 = (integralImg3.at<double>(Point(x+w,y+h)) - integralImg3.at<double>(Point(x+w,y)) - integralImg3.at<double>(Point(x,y+h)) + integralImg3.at<double>(Point(x,y)))/(max3+0.000001);
        sum4 = (integralImg4.at<double>(Point(x+w,y+h)) - integralImg4.at<double>(Point(x+w,y)) - integralImg4.at<double>(Point(x,y+h)) + integralImg4.at<double>(Point(x,y)))/(max4+0.000001);
        sum5 = (integralImg5.at<double>(Point(x+w,y+h)) - integralImg5.at<double>(Point(x+w,y)) - integralImg5.at<double>(Point(x,y+h)) + integralImg5.at<double>(Point(x,y)))/(max5+0.000001);
        sum6 = (integralImg6.at<double>(Point(x+w,y+h)) - integralImg6.at<double>(Point(x+w,y)) - integralImg6.at<double>(Point(x,y+h)) + integralImg6.at<double>(Point(x,y)))/(max6+0.000001);
        sum7 = (integralImg7.at<double>(Point(x+w,y+h)) - integralImg7.at<double>(Point(x+w,y)) - integralImg7.at<double>(Point(x,y+h)) + integralImg7.at<double>(Point(x,y)))/(max7+0.000001);
        sum8 = (integralImg8.at<double>(Point(x+w,y+h)) - integralImg8.at<double>(Point(x+w,y)) - integralImg8.at<double>(Point(x,y+h)) + integralImg8.at<double>(Point(x,y)))/(max8+0.000001);*/
        
       
        sum0 = (sum(integralImg0(bb))[0])/(max0+0.000001);
        sum1 = (sum(integralImg1(bb))[0])/(max1+0.000001);
        sum2 = (sum(integralImg2(bb))[0])/(max2+0.000001);
        sum3 = (sum(integralImg3(bb))[0])/(max3+0.000001);
        sum4 = (sum(integralImg4(bb))[0])/(max4+0.000001);
        sum5 = (sum(integralImg5(bb))[0])/(max5+0.000001);
        sum6 = (sum(integralImg6(bb))[0])/(max6+0.000001);
        sum7 = (sum(integralImg7(bb))[0])/(max7+0.000001);
        sum8 = (sum(integralImg8(bb))[0])/(max8+0.000001);
        
        // histogram of ith rectangle
        std::vector<float> totalSum = {static_cast<float>(sum0), static_cast<float>(sum1), static_cast<float>(sum2), static_cast<float>(sum3), static_cast<float>(sum4), static_cast<float>(sum5), static_cast<float>(sum6), static_cast<float>(sum7), static_cast<float>(sum8)};
        newSum.insert (newSum.end(), totalSum.begin(), totalSum.end());
    }
    return newSum;
}


// Train with xgBoost
//-------------------
BoosterHandle ImagePatch::trainTheDataXGBoost(std::vector<std::vector<float>> train, std::vector<float>label, int r, int c, int it) {
    BoosterHandle handle;
    DMatrixHandle h_train[1];
    XGDMatrixCreateFromMat((float *) &train[0], r, c, -1, &h_train[0]); //take from vector &positives[0]
    
    
    // load the labels
    XGDMatrixSetFloatInfo(h_train[0], "label", &label[0], r);
    
    // read back the labels, just a sanity check
    /*bst_ulong bst_result;
     const float *out_floats;
     XGDMatrixGetFloatInfo(h_train[0], "label" , &bst_result, &out_floats);
     for (unsigned int i=0;i<bst_result;i++)
     std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;*/
    
    // create the booster and load some parameters
    XGBoosterCreate(h_train, 1, &handle);
    XGBoosterSetParam(handle, "booster", "gbtree");
    XGBoosterSetParam(handle, "objective", "reg:linear");
    XGBoosterSetParam(handle, "max_depth", "2");
    XGBoosterSetParam(handle, "eta", "0.1");
    XGBoosterSetParam(handle, "min_child_weight", "1");
    XGBoosterSetParam(handle, "subsample", "0.5");
    XGBoosterSetParam(handle, "colsample_bytree", "1");
    XGBoosterSetParam(handle, "num_parallel_tree", "1");
    
    // perform 200 learning iterations
    for (int iter=0; iter<it; iter++)
        XGBoosterUpdateOneIter(handle, iter, h_train[0]);

    XGDMatrixFree(h_train[0]);
    return handle;
}


// destructor
//-----------
ImagePatch::~ImagePatch(){}