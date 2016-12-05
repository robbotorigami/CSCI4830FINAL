#pragma once

#include <opencv2/core/core.hpp>
/*
	Image meta data definition and helper functions for the caching system
*/
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


ImageMetaData* findByName(std::vector<ImageMetaData> &v, const char *fileName);
bool containedInGroup(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName);

void metaDataToFile(std::vector<ImageMetaData> &v, char *fileName);
void metaDataFromFile(std::vector<ImageMetaData> &v, char *fileName);

void groupsToFile(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName);
void groupsFromFile(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName, std::vector<ImageMetaData> &data);
