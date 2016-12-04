#include "ApplicationInterface.h"

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

using namespace std;
using namespace cv;

void ApplicationInterface::loadMetaData() {
	queue<char*> fileList;
	metaDataFromFile(imageList, META_DATA_CACHE_FILENAME);
	//-----------------GET LIST OF IMAGES-----------------------
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = FindFirstFileA(FOLDER_PATH "/*.jpg", &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("Bad path!\n");
		return;
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

	metaDataToFile(imageList, META_DATA_CACHE_FILENAME);
}

void ApplicationInterface::computeDuplicates() {
	//groupsFromFile(imageGroups, GROUPS_CACHE_FILENAME, imageList);
	//---------------NOW COMPARE IMAGES TO EACH OTHER------------------
	for (vector<ImageMetaData>::iterator i = imageList.begin(); i < imageList.end(); i++) {
		for (vector< vector<ImageMetaData*>* >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
			for (vector<ImageMetaData*>::iterator k = (*j)->begin(); k < (*j)->end(); k++) {
				if (duplicateDetect(*i, **k)) {
					(*j)->push_back(&*i);
					goto match_found;
				}
			}
		}
		{
			vector<ImageMetaData*>* dummy = new vector<ImageMetaData*>();
			dummy->push_back(&*i);
			imageGroups.push_back(dummy);
		}
	match_found:
		;
		printf("Complete: %d/%d\n", (i - imageList.begin()), imageList.size());
	}
	groupsToFile(imageGroups, GROUPS_CACHE_FILENAME);
}

void ApplicationInterface::displayGroups() {
	//---------------PRINT OUT THE GROUPS-----------------------
	for (vector< vector<ImageMetaData*>* >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
		if ((*j)->size() <= 1) continue;
		printf("GROUP UNDER: %s\n", (*(*j)->begin())->fileName);
		for (vector<ImageMetaData*>::iterator k = (*j)->begin() + 1; k < (*j)->end(); k++) {
			printf("\tITEM: %s\n", (*k)->fileName);
			namedWindow((*k)->fileName, WINDOW_AUTOSIZE);
			Mat dispImage;
			resize(imread((*k)->fileName), dispImage, (*k)->usedSize);
			imshow((*k)->fileName, dispImage);
		}
		namedWindow((*(*j)->begin())->fileName, WINDOW_AUTOSIZE);
		Mat dispImage;
		resize(imread((*(*j)->begin())->fileName), dispImage, (*(*j)->begin())->usedSize);
		imshow((*(*j)->begin())->fileName, dispImage);
		waitKey(0);
		destroyAllWindows();
	}
}

void ApplicationInterface::writeFileStructure() {

}