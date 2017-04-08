//
// Created by vladislav on 08.04.17.
//

#include "imfun.h"

bool showPointedImage(cv::Mat image, const vector<cv::Point2f>& points) {

    if(!image.data)                              
    {
        cout <<  "Could not show the image" << endl ;
        return false;
    }

    for(int i = 0; i < points.size(); ++i)
    {
        circle(image, points.at(i), 2, cv::Scalar(0, 255, 0), -1);
    }

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", image);

    cv::waitKey(0);                                          
    
    return true;
}

