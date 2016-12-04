#pragma once

#include <queue>
#include <vector>
#include <list>

#include "imageMetaData.h"

#define FOLDER_PATH "C:/Users/Cyborg9/Documents/PhotoSorter_images/"

#define testImage1 FOLDER_PATH "20160601_111929.jpg"
#define testImage2 FOLDER_PATH "0101.jpg"

#define META_DATA_CACHE_FILENAME "imageMetaData.dat"
#define GROUPS_CACHE_FILENAME "imageGroups.dat"


struct natureImage {
	char* fileName;
	double rank;
};

class ApplicationInterface {
private:
	std::vector<char*> fileList;

	std::vector<ImageMetaData> imageList;
	std::vector<std::vector<ImageMetaData*>*> imageGroups;
	std::list<natureImage> natureRank;
	std::vector<ImageMetaData*> classifiedList;
	char* folderPath;
	
public:
	ApplicationInterface() :folderPath(FOLDER_PATH)
	{}
	~ApplicationInterface() {
		for (std::vector< std::vector<ImageMetaData*>* >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
			delete *j;
		}
		for (std::vector<char*>::iterator j = fileList.begin(); j < fileList.end(); j++) {
			delete *j;
		}
	}

	void index();
	void setFolderPath(char* folderPath);
	void loadMetaData(bool dumpCache);
	void computeDuplicates(bool dumpCache);
	void displayGroups();
	void rankNatural();

	void cascadeClassify(std::string classifier);


};