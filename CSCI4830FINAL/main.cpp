#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <cstring>
#include <Windows.h>


#include "imageMetaData.h"
#include "CVUtils.h"

#define FOLDER_PATH "C:/Users/Cyborg9/Documents/PhotoSorter_images/"

#define testImage1 FOLDER_PATH "20160601_111931.jpg"
#define testImage2 FOLDER_PATH "20160601_112831.jpg"

#define META_DATA_CACHE_FILENAME "C:/Users/Cyborg9/Documents/PhotoSorter_images/imageMetaData.dat"
//#define DUMP_CACHE

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	queue<char*> fileList;
	vector<ImageMetaData> imageList;
#ifndef DUMP_CACHE
	metaDataFromFile(imageList, META_DATA_CACHE_FILENAME);
#endif
	//-----------------GET LIST OF IMAGES-----------------------
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = FindFirstFileA(FOLDER_PATH "/*.jpg", &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("Bad path!\n");
		return -1;
	}

	do {
		char* fileName = reinterpret_cast<char*>(
			malloc(strlen(FOLDER_PATH) + strlen(reinterpret_cast<char*>(ffd.cFileName)))
			);
		strcpy(fileName, FOLDER_PATH);
		strcat(fileName, reinterpret_cast<char*>(ffd.cFileName));
		fileList.push(fileName);
		//printf("FILENAME: %s\n", fileName);
	} while (FindNextFileA(hFind, &ffd) != 0);

	FindClose(hFind);

	//---------------USED CACHED FEATURES, OR DETECT NEW--------------
	while (fileList.size() > 0) {
		char *fileName = fileList.front();
		bool neededToLoad = true;
		for (vector<ImageMetaData>::iterator i = imageList.begin(); i < imageList.end(); i++) {
			neededToLoad = (strcmp(fileName, i->fileName) != 0);
			if (!neededToLoad) break;
		}
		if (neededToLoad) {
			ImageMetaData imd = computeDescriptors(fileName);
			Mat outImage;
			Mat inImage;
			resize(imread(imd.fileName), inImage, imd.usedSize);
			drawKeypoints(inImage, imd.keypoints, outImage);
			namedWindow("feature detect", WINDOW_AUTOSIZE);
			imshow("feature detect", outImage);
			waitKey(1);
			imageList.push_back(imd);
		}
		fileList.pop();
	}

	//Cache that data!
	metaDataToFile(imageList, META_DATA_CACHE_FILENAME);

	//ImageMetaData* imd1 = findByName(imageList, testImage1);
	//ImageMetaData* imd2 = findByName(imageList, testImage2);
	//if (imd1 == NULL || imd2 == NULL) {
	//	printf("Image not found!\n");
	//	return -1;
	//}
	//duplicateDetect(*imd1, *imd2);
	//return 0;

	//---------------NOW COMPARE IMAGES TO EACH OTHER------------------
	vector< vector<ImageMetaData> > imageGroups;
	for (vector<ImageMetaData>::iterator i = imageList.begin(); i < imageList.end(); i++) {
		for (vector< vector<ImageMetaData> >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
			for (vector<ImageMetaData>::iterator k = j->begin(); k < j->end(); k++) {
				if (duplicateDetect(*i, *k)) {
					(*j).push_back(*i);
					goto match_found;
				}
			}
		}
		{	
			vector<ImageMetaData> dummy;
			dummy.push_back(*i);
			imageGroups.push_back(dummy);
		}
	match_found:
		printf("Percent Complete: %f%%\n", 100*static_cast<float>((i-imageList.begin()))/imageList.size());
	}

	//---------------PRINT OUT THE GROUPS-----------------------
	for (vector< vector<ImageMetaData> >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
		printf("GROUP UNDER: %s\n", (*j->begin()).fileName);
		for (vector<ImageMetaData>::iterator k = j->begin()+1; k < j->end(); k++) {
			printf("\tITEM: %s\n", (*k).fileName);
		}
	}
	system("PAUSE");
	return 0;
}