#pragma once

#include <queue>
#include <vector>

#include "imageMetaData.h"

#define FOLDER_PATH "C:/Users/Cyborg9/Documents/PhotoSorter_images/"

#define testImage1 FOLDER_PATH "20160601_111929.jpg"
#define testImage2 FOLDER_PATH "0101.jpg"

#define META_DATA_CACHE_FILENAME FOLDER_PATH "imageMetaData.dat"
#define GROUPS_CACHE_FILENAME FOLDER_PATH "imageGroups.dat"

class ApplicationInterface {
private:
	std::vector<ImageMetaData> imageList;
	std::vector<std::vector<ImageMetaData*>*> imageGroups;
	
public:
	ApplicationInterface() {}
	~ApplicationInterface() {
		for (std::vector< std::vector<ImageMetaData*>* >::iterator j = imageGroups.begin(); j < imageGroups.end(); j++) {
			delete *j;
		}
	}

	void loadMetaData();
	void computeDuplicates();
	void displayGroups();
	void writeFileStructure();

};