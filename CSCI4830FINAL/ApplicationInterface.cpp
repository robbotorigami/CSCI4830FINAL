#include "ApplicationInterface.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
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


void ApplicationInterface::setFolderPath(char *folderPath) {
	this->folderPath = folderPath;
}

//Loop through all of the files in the path, and add them to the file list
void ApplicationInterface::index() {
	//-----------------GET LIST OF IMAGES-----------------------
	WIN32_FIND_DATAA ffd;
	char formatSpecifier[100];
	strcpy(formatSpecifier, folderPath);
	strcat(formatSpecifier, "/*.jpg");
	HANDLE hFind = FindFirstFileA(formatSpecifier, &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("Bad path!\n");
		return;
	}

	do {
		char* fileName = reinterpret_cast<char*>(
			malloc(strlen(folderPath) + strlen(reinterpret_cast<char*>(ffd.cFileName)))
			);
		strcpy(fileName, folderPath);
		strcat(fileName, reinterpret_cast<char*>(ffd.cFileName));
		fileList.push_back(fileName);
		printf("FILENAME: %s\n", fileName);
	} while (FindNextFileA(hFind, &ffd) != 0);

	FindClose(hFind);
}

//Goes through all files, and computes keypoints and descriptors for uncached values
void ApplicationInterface::loadMetaData(bool dumpCache) {
	queue<char*> fileList;
	for (vector<char*>::iterator i = this->fileList.begin(); i < this->fileList.end(); i++) fileList.push(*i);
	char cachePath[100];
	strcpy(cachePath, folderPath);
	strcat(cachePath, META_DATA_CACHE_FILENAME);
	if(!dumpCache) metaDataFromFile(imageList, cachePath);

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
	destroyAllWindows();
	metaDataToFile(imageList, cachePath);
}

//Goes through all files, and sorts any uncached images
void ApplicationInterface::computeDuplicates(bool dumpCache) {
	char cachePath[100];
	strcpy(cachePath, folderPath);
	strcat(cachePath, GROUPS_CACHE_FILENAME);
	if (!dumpCache) groupsFromFile(imageGroups, cachePath, imageList);
	//---------------NOW COMPARE IMAGES TO EACH OTHER------------------
	for (vector<ImageMetaData>::iterator i = imageList.begin(); i < imageList.end(); i++) {
		if (containedInGroup(imageGroups, (*i).fileName)) continue; //Skip adding any that have been cached
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
	destroyAllWindows();
	groupsToFile(imageGroups, cachePath);
}

//Display the images in each bin in bulk
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

bool rank_nm(natureImage nm1, natureImage nm2) {
	return nm1.rank > nm2.rank;
}

//Computes the "Naturalness" of each image
void ApplicationInterface::rankNatural() {
	for (vector<char*>::iterator i = fileList.begin(); i < fileList.end(); i++) {
		natureImage nm;
		nm.fileName = *i;
		nm.rank = computeNatureRank(*i);
		natureRank.push_back(nm);
	}
	natureRank.sort(rank_nm);
}

//Attempts to classify images using a cascadeclassifier
void ApplicationInterface::cascadeClassify(string classifier) {
	//-----------CLEAR ALL FILES------------
	WIN32_FIND_DATAA ffd;
	char formatSpecifier[100];
	strcpy(formatSpecifier, folderPath);
	strcat(formatSpecifier, "/classified/*.jpg");
	HANDLE hFind = FindFirstFileA(formatSpecifier, &ffd);

	if (INVALID_HANDLE_VALUE != hFind)
	{
		do {
			char* fileName = reinterpret_cast<char*>(
				malloc(strlen(folderPath) + strlen(reinterpret_cast<char*>(ffd.cFileName)))
				);
			strcpy(fileName, folderPath);
			strcat(fileName, "classified/");
			strcat(fileName, reinterpret_cast<char*>(ffd.cFileName));
			DeleteFileA(fileName);
		} while (FindNextFileA(hFind, &ffd) != 0);
		FindClose(hFind);
	}

	CascadeClassifier cascade;
	classifier = folderPath + classifier;
	if (!cascade.load(classifier)) {
		printf("Bad classifier file: %s\n", classifier);
		return;
	}
	int j = 0;
	for (vector<char*>::iterator i = fileList.begin(); i < fileList.end(); i++) {
		if (classifyImage((*i), cascade)) {
			char fileName[200];
			strcpy(fileName, folderPath);
			strcat(fileName, "classified/");
			sprintf(fileName + strlen(fileName), "%d.jpg", j);
			CopyFileA((*i), fileName, false);
			j++;
		}
		printf("Complete: %d/%d\n", (i - fileList.begin()), fileList.size());
	}
}


//Places the natural ranks into a file
void ApplicationInterface::writeOutRanks() {
	//-----------CLEAR ALL FILES------------
	WIN32_FIND_DATAA ffd;
	char formatSpecifier[100];
	strcpy(formatSpecifier, folderPath);
	strcat(formatSpecifier, "/ranked/*.jpg");
	HANDLE hFind = FindFirstFileA(formatSpecifier, &ffd);

	if (INVALID_HANDLE_VALUE != hFind)
	{
		do {
			char* fileName = reinterpret_cast<char*>(
				malloc(strlen(folderPath) + strlen(reinterpret_cast<char*>(ffd.cFileName)))
				);
			strcpy(fileName, folderPath);
			strcat(fileName, "ranked/");
			strcat(fileName, reinterpret_cast<char*>(ffd.cFileName));
			DeleteFileA(fileName);
		} while (FindNextFileA(hFind, &ffd) != 0);
		FindClose(hFind);
	}

	namedWindow("disp");
	int j = 0;
	for (list<natureImage>::iterator i = natureRank.begin(); i != natureRank.end(); i++) {
		char fileName[200];
		strcpy(fileName, folderPath);
		strcat(fileName, "ranked/");
		sprintf(fileName + strlen(fileName), "%d.jpg", j);
		imshow("disp", imread((*i).fileName));
		CopyFileA((*i).fileName, fileName, false);
		waitKey(500);
		j++;
	}
	destroyAllWindows();
}