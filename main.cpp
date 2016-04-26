//
//  main.cpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//


#include "Rectangles.hpp"
#include "ImagePatch.hpp"
#include "Image.hpp"
//#include "opencv2/ml/ml.hpp"
//#include "xgboost/c_api.h"
//#include <dmlc/timer.h>

#include <iostream>

using namespace std;
using namespace cv;


// to view the matrix
//-------------------
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


int main(int argc, const char * argv[]) {
    
    // call constructor of Rectangles to initialise the position of the rectangles as boundingBox(sR, sC, width, height)
    //------------------------------------------------------------------------------------------------------------------
    //const int patchSize{41};
    //const int nbRectangle{10000};
    
    Rectangles rect;
    cv::vector<cv::Rect> bBox = rect.generate_rectangles();
    
    int windowWidth = rect.get_patchWidth();
    int windowHeight = rect.get_patchHeight();
    
    
    // call constructor of ImagePatch to initialise the threshold and the number of rectangles used for the orientation feature representation
    //----------------------------------------------------------------------------------------------------------------------------------------
    
    ImagePatch feature;

    // read data from folder and make trainData and trainLabel
    auto t0 = chrono::high_resolution_clock::now();
    String f0 = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/0/";
    vector<String> filenames0;
    glob(f0, filenames0);
    auto t1 = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(t1-t0).count() << " msec\n";

    //cout << filenames0.size() << endl;

    //String folder = "/Users/barbara/Documents/MATLAB/retinal_dataset/";
    //cv::String f = "/Users/barbara/Dropbox/Image/";
    
    vector<vector<float>> negatives;
    vector<float> labelNegative;
    int label0 = 0;
    auto t2 = chrono::high_resolution_clock::now();
    negatives = feature.extract_features_of_patches(bBox, filenames0);
    auto t3 = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::seconds>(t3-t2).count() << " sec for making feature vector 0\n";

    labelNegative = feature.extract_label_of_patches(label0);
    int nbRowNeg = (int)feature.get_SizeFileName();
   
    
    String f1 = "/Users/otl/Documents/MATLAB/Patient1/newImagePatches/1/";
    vector<String> filenames1;
    glob(f1, filenames1);
    cout << filenames1.size() << endl;
   
    vector<vector<float>> positives;
    vector<float> labelPositive;
    int label1 = 1;
    auto t4 = chrono::high_resolution_clock::now();
    positives = feature.extract_features_of_patches(bBox, filenames1);
    auto t5 = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(t5-t4).count() << " msec for making feature vector 1\n";
    
    labelPositive = feature.extract_label_of_patches(label1);
    
    positives.insert(positives.end(), negatives.begin(), negatives.end());
    labelPositive.insert(labelPositive.end(), labelNegative.begin(), labelNegative.end());
    
    
    int nbRectPart = feature.get_NbRectangles();
    int nbRowPos = (int)feature.get_SizeFileName();
    //int nbRow = nbRowNeg + nbRowPos;
    int nbRow = 2000;
    int nbCols = nbRectPart*9;
    
    
    // prepare for xgboost
    BoosterHandle h_booster;
    int iterations = 200;
    h_booster = feature.trainTheDataXGBoost(positives, labelPositive, nbRow, nbCols, iterations);
    
    
    // free xgboost internal structures
    XGBoosterFree(h_booster);
    
    
    
    // call constructor of Image to initialise the stride and the reduction parameter needed for the sliding window
    //-------------------------------------------------------------------------------------------------------------
    Image slidingWindow;
    int thresh = feature.get_ThMagnitude();
    int rectangleNb = feature.get_NbRectangles();
    
    // read whole images from folder and make trainData and trainLabel
    String f3 = "/Users/otl/Documents/MATLAB/Patient1/NewImages/";
    vector<String> filenamesImg;
    glob(f3, filenamesImg);
    cout << filenamesImg.size() << endl;
    Mat img = imread(filenamesImg[0]);
    slidingWindow.set_imageHeight(img.rows);
    slidingWindow.set_imageWidth(img.cols);
    slidingWindow.set_hBooster(h_booster);
    
    /*
    stringstream ss;
    string folderName = "0";
    string folderCreateComand = "mkdir " + folderName;
    
    system(folderCreateComand.c_str());
    
    
    string name = "patient1";
    string type = ".jpg";
    */
        
    auto t10 = chrono::high_resolution_clock::now();
    //for(size_t i = 0; i < filenamesImg.size(); ++i)
    for(size_t i = 0; i < 1; ++i)
    {
        Mat img = imread(filenamesImg[i]);
        
        Mat map = slidingWindow.downscale_image(windowWidth, windowHeight, img, thresh, rectangleNb);
        
        /*
        ss << folderName << "/" << name << (i + 1) << type;
        string filename = ss.str();
        ss.str("");
        
        //imwrite( "/Users/otl/Documents/MATLAB/Patient1/NewImages/Gray_Image.png", map);
        //imwrite( format("folder/image%d.png", i ), img);
        namedWindow( "MAP", CV_WINDOW_AUTOSIZE );
        imshow( "MAP", map);
        waitKey(0);
        
        */

    }
    auto t11 = chrono::high_resolution_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(t11-t10).count() << " msec for sliding window\n";


        
    // read images
    
    return 0;
}
