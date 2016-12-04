#pragma once

#include <opencv2/core/core.hpp>
#include <stdlib.h>
#include <vector>
#include <fstream>

struct ImageMetaData {
	char fileName[100];
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::Size usedSize;

	void toFile(std::ofstream &fout);
	void fromFile(std::ifstream &fin);
};

void metaDataToFile(std::vector<ImageMetaData> &v, char *fileName);
void metaDataFromFile(std::vector<ImageMetaData> &v, char *fileName);
