//
//  ImagePatch.cpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//

#include "ImagePatch.hpp"
#include "Rectangles.hpp"
#include <iostream>
#include <iterator>
#include <chrono>

using namespace cv;


// constructors
//-------------
ImagePatch::ImagePatch():thresholdMagnitude{3}, partialRectangleNB{1111} {}

ImagePatch::ImagePatch(int th, int nbRect):thresholdMagnitude{th}, partialRectangleNB{nbRect} {}


// to view the matrix
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


// extract the features from the image patches
//--------------------------------------------
void ImagePatch::extract_features_of_patches(std::vector<std::array<int, 4>> bb) {
    bbRectangle = bb;
    
    //String folder = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/1/";
    //String folder = "/Users/barbara/Documents/MATLAB/retinal_dataset/";
    String folder = "/Users/otl/Dropbox/Image/";
    vector<String> filenames;
    glob(folder, filenames);
    
    std::vector<std::vector<double>> orientationFeatures;
    
    //for(size_t i = 0; i < filenames.size(); ++i)
    for(size_t i = 0; i < 2; ++i)
    {
        image = imread(filenames[i]);
        
        if(!image.data)
            std::cerr << "Problem loading image!!!" << std::endl;
        
        orientationFeatures = make_orientationHistogramFeatures();
    }
    //printet ganze zahlen!!!!!!!!!!!!!!!
    print(orientationFeatures, partialRectangleNB, 9);
    

}


// calculate the orientation histogram features
//---------------------------------------------
std::vector<std::vector<double>> ImagePatch::make_orientationHistogramFeatures() {
    
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
    threshold(magnitude, orient0, thresholdMagnitude, max_BINARY_value, THRESH_BINARY);
    
    // group to orientations
    Mat orient1 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient2 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient3 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient4 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient5 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient6 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient7 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    Mat orient8 = Mat::zeros(direction.rows, direction.cols, CV_32F);
    
    for(int i=0; i<direction.rows; i++) {
        float* dir = direction.ptr<float>(i);
        
        float* row1 = orient1.ptr<float>(i);
        float* row2 = orient2.ptr<float>(i);
        float* row3 = orient3.ptr<float>(i);
        float* row4 = orient4.ptr<float>(i);
        float* row5 = orient5.ptr<float>(i);
        float* row6 = orient6.ptr<float>(i);
        float* row7 = orient7.ptr<float>(i);
        float* row8 = orient8.ptr<float>(i);
        
        for(int j=0; j<direction.cols; j++) {
            
            float value = dir[j];
            
            if (value >= 0 && value < 45)
                row1[j] = 1;
            if (value >= 45 && value < 90)
                row2[j] = 1;
            if (value >= 90 && value < 135)
                row3[j] = 1;
            if (value >= 135 && value < 180)
                row4[j] = 1;
            if (value >= 180 && value < 225)
                row5[j] = 1;
            if (value >= 225 && value < 270)
                row6[j] = 1;
            if (value >= 270 && value < 315)
                row7[j] = 1;
            if (value >= 315 && value < 360)
                row8[j] = 1;
            
            if (value < 0 || value >=360)
                std::cout << "there is a mistake! degrees from [0, 360)" << std::endl;
        }
    }
    
    Mat integralImg0, integralImg1, integralImg2, integralImg3, integralImg4, integralImg5, integralImg6, integralImg7, integralImg8;
    Mat edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8;
    
    integral(orient0, integralImg0);
    
    multiply(orient0, orient1, edge1, 1, -1);
    integral(edge1, integralImg1);
  
    multiply(orient0, orient2, edge2, 1, -1);
    integral(edge2, integralImg2);
    
    multiply(orient0, orient3, edge3, 1, -1);
    integral(edge3, integralImg3);
    
    multiply(orient0, orient4, edge4, 1, -1);
    integral(edge4, integralImg4);
    
    multiply(orient0, orient5, edge5, 1, -1);
    integral(edge5, integralImg5);
    
    multiply(orient0, orient6, edge6, 1, -1);
    integral(edge6, integralImg6);
    
    multiply(orient0, orient7, edge7, 1, -1);
    integral(edge7, integralImg7);
    
    multiply(orient0, orient8, edge8, 1, -1);
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
    // look what is faster .at or this
    double max0, max1, max2, max3, max4, max5, max6, max7, max8;
    
    // measure time
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::minMaxIdx(integralImg0, nullptr, &max0);
    cv::minMaxIdx(integralImg1, nullptr, &max1);
    cv::minMaxIdx(integralImg2, nullptr, &max2);
    cv::minMaxIdx(integralImg3, nullptr, &max3);
    cv::minMaxIdx(integralImg4, nullptr, &max4);
    cv::minMaxIdx(integralImg5, nullptr, &max5);
    cv::minMaxIdx(integralImg6, nullptr, &max6);
    cv::minMaxIdx(integralImg7, nullptr, &max7);
    cv::minMaxIdx(integralImg8, nullptr, &max8);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "msec\n";
    
    // measure time
    t0 = std::chrono::high_resolution_clock::now();
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

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "msec\n";

    
    
    //std::cout << integralImg0 << std::endl;
    
    std::vector<double> newSum {};
    std::vector<std::vector<double>> totalFeaturePerPatch;

    // calculate the region sum in a rectangle of each map
    for (int i = 0; i != partialRectangleNB; ++i) {
        int x = bbRectangle[i][1];
        int y = bbRectangle[i][2];
        int w = bbRectangle[i][3];
        int h = bbRectangle[i][4];
        //Rect rect(sR, sC, width, height);
        
        //std::cout << "x " << x << ", y " << y << ", w " << w << ", h " << h <<std::endl;
        //std::cout << integralImg0.depth() << std::endl;
        
        double sum0 = (integralImg0.at<double>(Point(x+w,y+h)) - integralImg0.at<double>(Point(x+w,y)) - integralImg0.at<double>(Point(x,y+h)) + integralImg0.at<double>(Point(x,y)))/(max0+0.000001);
        double sum1 = (integralImg1.at<double>(Point(x+w,y+h)) - integralImg1.at<double>(Point(x+w,y)) - integralImg1.at<double>(Point(x,y+h)) + integralImg1.at<double>(Point(x,y)))/(max1+0.000001);
        double sum2 = (integralImg2.at<double>(Point(x+w,y+h)) - integralImg2.at<double>(Point(x+w,y)) - integralImg2.at<double>(Point(x,y+h)) + integralImg2.at<double>(Point(x,y)))/(max2+0.000001);
        double sum3 = (integralImg3.at<double>(Point(x+w,y+h)) - integralImg3.at<double>(Point(x+w,y)) - integralImg3.at<double>(Point(x,y+h)) + integralImg3.at<double>(Point(x,y)))/(max3+0.000001);
        double sum4 = (integralImg4.at<double>(Point(x+w,y+h)) - integralImg4.at<double>(Point(x+w,y)) - integralImg4.at<double>(Point(x,y+h)) + integralImg4.at<double>(Point(x,y)))/(max4+0.000001);
        double sum5 = (integralImg5.at<double>(Point(x+w,y+h)) - integralImg5.at<double>(Point(x+w,y)) - integralImg5.at<double>(Point(x,y+h)) + integralImg5.at<double>(Point(x,y)))/(max5+0.000001);
        double sum6 = (integralImg6.at<double>(Point(x+w,y+h)) - integralImg6.at<double>(Point(x+w,y)) - integralImg6.at<double>(Point(x,y+h)) + integralImg6.at<double>(Point(x,y)))/(max6+0.000001);
        double sum7 = (integralImg7.at<double>(Point(x+w,y+h)) - integralImg7.at<double>(Point(x+w,y)) - integralImg7.at<double>(Point(x,y+h)) + integralImg7.at<double>(Point(x,y)))/(max7+0.000001);
        double sum8 = (integralImg8.at<double>(Point(x+w,y+h)) - integralImg8.at<double>(Point(x+w,y)) - integralImg8.at<double>(Point(x,y+h)) + integralImg8.at<double>(Point(x,y)))/(max8+0.000001);
        
        // histogram of ith rectangle
        std::vector<double> totalSum = {sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8};
        newSum.insert (newSum.end(), totalSum.begin(), totalSum.end());

        
    }
    
    totalFeaturePerPatch.push_back(newSum);
    
    
    //printet ganze zahlen!!!!!!!!!!!!!!!
    print(totalFeaturePerPatch, partialRectangleNB, 9);
    

    return totalFeaturePerPatch;
    
}

// destructor
//-----------
ImagePatch::~ImagePatch(){}