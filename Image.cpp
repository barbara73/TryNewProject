//
//  Image.cpp
//  TryNewProject
//
//  Created by Barbara Jesacher on 28.03.16.
//  Copyright (c) 2016 otl. All rights reserved.
//

#include "Image.hpp"
//#include "Rectangles.hpp"
#include "ImagePatch.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

// to view the matrix
template <class T>
void print(T & t, size_t rows, size_t columns)
{
    for(size_t i = 0;i < rows; ++i)
    {
        for(size_t j = 0;j < columns; ++j)
            printf("%f ", t[i][j]);
        
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

// convert vector to cv::mat
//--------------------------
Mat vectorToMat(vector<vector<float>> vec) {
    
    // Create a new, _empty_ cv::Mat with the row size of positives
    Mat mat(0, (int)vec[0].size(), DataType<float>::type);
    for (auto i = 0; i < vec.size(); ++i) {
        // Make a temporary cv::Mat row and add to mat _without_ data copy
        Mat sample(1, (int)vec[0].size(), DataType<float>::type, vec[i].data());
        mat.push_back(sample);
    }
    return mat;
}


// constructor initialising number of reduction, max stride and reduction by param
//--------------------------------------------------------------------------------
Image::Image(): nbReductions{3}, stride{15}, reductionParameter{0.85} {}
Image::Image(int red, int str, float redParam):nbReductions{red}, stride{str}, reductionParameter{redParam} {}


// downscale image
//----------------
Mat Image::downscale_image(int wWidth, int wHeight, Mat img, int th, int rectNb) {
    blockWidth = wWidth;
    blockHeight = wHeight;
    Mat paddedImg, downscaledImg, resizedMap, map, tempMap, tempImg, pImg;
    int w = (blockWidth-1)/2;
    int h = (blockHeight-1)/2;
    int borderType = BORDER_CONSTANT;
    newMap = Mat::zeros(imgHeight, imgWidth, CV_32F);
    tempMap = Mat::zeros(imgHeight, imgWidth, CV_32F);
    vector<vector<float>> vectorMap;
    
    
    downscaledImg = img;
    

    auto t0 = chrono::high_resolution_clock::now();
    for (int i=0; i!=nbReductions; ++i) {
    
        copyMakeBorder(downscaledImg, pImg, h, h, w, w, borderType, 0);    //padd image with zeros
        paddedImg = pImg;
        
        auto t2 = chrono::high_resolution_clock::now();
        
        // slide window over downscaled image
        vectorMap = slide_window_over_image(i, paddedImg, th, rectNb);
       
        map.release();
        map = vectorToMat(vectorMap);       //make Mat image which is smaller than before
        vector<vector<float>> newVector;
        vectorMap.swap(newVector);
        
      
        //namedWindow( "MAP", CV_WINDOW_AUTOSIZE );
        //imshow( "MAP", map);
        //waitKey(0);
        
        const Size size(imgWidth, imgHeight);
        resize(map, resizedMap, size);      // resize map to original size WHY NOT WORKING???????
        cout << "map: " << map.size() << "resized Map: " << resizedMap.size() << endl;
        
        //namedWindow( "resizedMAP", CV_WINDOW_AUTOSIZE );
        //imshow( "resizedMAP", resizedMap);
        //waitKey(0);
        
        max(resizedMap, newMap, tempMap);   // take maximum pixel of each map
        newMap = tempMap;
        
        auto t3 = chrono::high_resolution_clock::now();
        cout << chrono::duration_cast<chrono::seconds>(t3-t2).count() << " sec for sliding window per image\n";
        
        resize(downscaledImg, tempImg, Size(), reductionParameter, reductionParameter, CV_INTER_AREA);          //ok
        downscaledImg = tempImg;
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::seconds>(t1-t0).count() << " sec for all images\n";
    

    return newMap;
}


// slide window over image
//------------------------
vector<vector<float>> Image::slide_window_over_image(int param, Mat img, int thresh, int nbRectangles) {
    stride -= param;
    
    vector<vector<float>> score(img.rows-blockHeight+1, vector<float>(img.cols-blockWidth+1));
    
    //int count = 0;
    //int i = round(img.rows+stride-1)/stride;
    //int j = round(img.cols+stride-1)/stride;
    //vector<int> rowList(i*j);
    //vector<int> colList(i*j);
    
    //vector<float> orientationFeatures(i*j);
    //vector<std::vector<float>> totalFeaturePerPatch;
    
    for (auto v : score) {
        for (auto s : v) {
            s = 0.;
        }
    }
    
    for (int r=0; r<img.rows-blockHeight; r=r+stride) {
        for (int c=0; c<img.cols-blockWidth; c=c+stride) {
            //rowList[count] = r;
            //colList[count] = c;
            
            //cout << "r: " << r << ", c: " << c << endl;
            //cout << "stride: " << stride << endl;
            
            Rect roi(c, r, blockWidth, blockHeight);
            Mat imageRoi = img(roi);
            
            Scalar meanValue = mean(imageRoi);
            
            
             if (meanValue[0]>20) {
             std::vector<float> orientationFeatures;
             //std::vector<std::vector<float>> totalFeaturePerPatch;
             orientationFeatures = ImagePatch::make_orientationHistogramFeatures(imageRoi, bb, thresh, nbRectangles);
             float score = testTheDataXGBoost(classifier, orientationFeatures, nbCols);

             }
            /*
            if (meanValue[0] > 20) {
                score[r][c] = 0.8;
            }
            */
            
        }
        
    }

    //totalFeaturePerPatch.push_back(orientationFeatures);
    
    return score;
}
    
// Test data with xgboost
//-----------------------
float Image::testTheDataXGBoost(BoosterHandle handle, vector<float> test, int r, int c) {
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float *) &test[0], r, c, -1, &h_test);
    bst_ulong out_len;
    const float *f;
    XGBoosterPredict(handle, h_test, 0,0,&out_len,&f);

    for (unsigned int i=0;i<out_len;i++)
        std::cout << "prediction[" << i << "]=" << f[i] << std::endl;
    XGDMatrixFree(h_test);
    return f[0];
}

// destructor
Image::~Image() {}