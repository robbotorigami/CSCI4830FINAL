#pragma once
#include <opencv2/objdetect/objdetect.hpp>

#include "imageMetaData.h"

ImageMetaData computeDescriptors(char* fileName);

bool* precomputeHistogramMatches(std::vector<ImageMetaData> &v);
bool duplicateDetect(ImageMetaData &imd1, ImageMetaData &imd2);

double computeNatureRank(char* fname);

bool classifyImage(char* fname, cv::CascadeClassifier &classifier);
