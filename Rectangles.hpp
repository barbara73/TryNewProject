//
//  Rectangles.hpp
//  TryNewProject
//
//  Created by otl on 21/03/16.
//  Copyright © 2016 otl. All rights reserved.
//

#ifndef Rectangles_hpp
#define Rectangles_hpp

#include <stdio.h>
#include <vector>
#include <array>

class Rectangles {
    int patchWidth;
    int patchHeight;
    int numberOfRectangles;
    
public:
    std::vector<std::array<int, 4>> generate_rectangles();
    
    Rectangles();
    Rectangles(int, int);
    Rectangles(int, int, int);
    ~Rectangles();
};


#endif /* Rectangles_hpp */
