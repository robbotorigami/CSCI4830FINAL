#pragma once
#include "imageMetaData.h"

ImageMetaData computeDescriptors(char* fileName);
ImageMetaData* findByName(std::vector<ImageMetaData> &v, const char *fileName);

bool* precomputeHistogramMatches(std::vector<ImageMetaData> &v);
bool duplicateDetect(ImageMetaData &imd1, ImageMetaData &imd2);
