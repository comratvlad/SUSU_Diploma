//
// Created by vladislav on 08.04.17.
//

#ifndef DATALOADPROJECT_IMFUN_H
#define DATALOADPROJECT_IMFUN_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;

bool showPointedImage(cv::Mat image, const vector<cv::Point2f>& points);
bool showPointedImage(const string& image_path, const string& points_path);


#endif //DATALOADPROJECT_IMFUN_H
