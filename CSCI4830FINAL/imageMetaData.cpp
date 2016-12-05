#include "imageMetaData.h"
#include <cstring>

using namespace cv;
using namespace std;

//Write out all of the data in a binary format
void ImageMetaData::toFile(ofstream &fout) {
	fout.write(fileName, sizeof(fileName));

	fout.write(reinterpret_cast<char*>(&descriptors.rows), sizeof(int));
	fout.write(reinterpret_cast<char*>(&descriptors.cols), sizeof(int));
	for (int i = 0; i < descriptors.rows; i++) {
		for (int j = 0; j < descriptors.cols; j++) {
			unsigned char value = descriptors.at<unsigned char>(i, j);
			fout.write(reinterpret_cast<char*>(&value), sizeof(unsigned char));
		}
	}

	size_t keypointsLength = keypoints.size();
	fout.write(reinterpret_cast<char*>(&keypointsLength), sizeof(size_t));
	for (vector<KeyPoint>::iterator i = keypoints.begin(); i < keypoints.end(); i++) {
		fout.write(reinterpret_cast<char*>(&(i->pt.x)), sizeof(float));
		fout.write(reinterpret_cast<char*>(&(i->pt.y)), sizeof(float));
		fout.write(reinterpret_cast<char*>(&(i->size)), sizeof(float));
		fout.write(reinterpret_cast<char*>(&(i->angle)), sizeof(float));
		fout.write(reinterpret_cast<char*>(&(i->response)), sizeof(float));
		fout.write(reinterpret_cast<char*>(&(i->octave)), sizeof(int));
		fout.write(reinterpret_cast<char*>(&(i->class_id)), sizeof(int));
	}

	fout.write(reinterpret_cast<char*>(&usedSize.width), sizeof(int));
	fout.write(reinterpret_cast<char*>(&usedSize.height), sizeof(int));	
}

//Read in the data from a binary file
void ImageMetaData::fromFile(ifstream &fin) {
	fin.read(fileName, sizeof(fileName));

	int rows, cols;
	fin.read(reinterpret_cast<char*>(&rows), sizeof(int));
	fin.read(reinterpret_cast<char*>(&cols), sizeof(int));
	descriptors = Mat(rows, cols, CV_8UC1);
	for (int i = 0; i < descriptors.rows; i++) {
		for (int j = 0; j < descriptors.cols; j++) {
			unsigned char value;
			fin.read(reinterpret_cast<char*>(&value), sizeof(unsigned char));
			descriptors.at<unsigned char>(i, j) = value;
		}
	}

	size_t keypointsLength;
	fin.read(reinterpret_cast<char*>(&keypointsLength), sizeof(size_t));
	keypoints = vector<KeyPoint>();
	for (size_t i = 0; i < keypointsLength; i++) {
		float x, y, size, angle, response;
		int octave, class_id;
		fin.read(reinterpret_cast<char*>(&x), sizeof(float));
		fin.read(reinterpret_cast<char*>(&y), sizeof(float));
		fin.read(reinterpret_cast<char*>(&size), sizeof(float));
		fin.read(reinterpret_cast<char*>(&angle), sizeof(float));
		fin.read(reinterpret_cast<char*>(&response), sizeof(float));
		fin.read(reinterpret_cast<char*>(&octave), sizeof(int));
		fin.read(reinterpret_cast<char*>(&class_id), sizeof(int));
		keypoints.push_back(KeyPoint(x, y, size, angle, response, octave, class_id));
	}
	int width, height;
	fin.read(reinterpret_cast<char*>(&width), sizeof(int));
	fin.read(reinterpret_cast<char*>(&height), sizeof(int));
	usedSize = Size(width, height);
}

//Loop over everything and cache it
void metaDataToFile(vector<ImageMetaData> &v, char* fileName) {
	ofstream fout(fileName, ios::out | ios::binary);
	size_t size = v.size();
	fout.write(reinterpret_cast<char*>(&size), sizeof(size_t));
	for (vector<ImageMetaData>::iterator i = v.begin(); i < v.end(); i++) {
		i->toFile(fout);
	}
}

void metaDataFromFile(vector<ImageMetaData> &v, char* fileName) {
	ifstream fin(fileName, ios::in | ios::binary);
	size_t size;
	fin.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	for (size_t i = 0; i < size; i++) {
		ImageMetaData imd;
		imd.fromFile(fin);
		v.push_back(imd);
	}
}

ImageMetaData* findByName(std::vector<ImageMetaData> &v, const char *fileName) {
	for (vector<ImageMetaData>::iterator i = v.begin(); i < v.end(); i++) {
		if (strcmp(i->fileName, fileName) == 0) {
			return &*i;
		}
	}
	return NULL;
}
bool containedInGroup(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName) {
	for (vector< vector<ImageMetaData*>* >::iterator j = v.begin(); j < v.end(); j++) {
		for (vector<ImageMetaData*>::iterator k = (*j)->begin(); k < (*j)->end(); k++) {
			if (strcmp((*k)->fileName, fileName) == 0) return true;
		}
	}
	return false;
}


void groupsToFile(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName) {
	ofstream fout(fileName, ios::out | ios::binary);
	size_t size = v.size();
	fout.write(reinterpret_cast<char*>(&size), sizeof(size_t));
	for (vector<vector<ImageMetaData*>*>::iterator i = v.begin(); i < v.end(); i++) {
		size = (*i)->size();
		fout.write(reinterpret_cast<char*>(&size), sizeof(size_t));
		for (vector<ImageMetaData*>::iterator j = (*i)->begin(); j < (*i)->end(); j++) {
			fout.write((*j)->fileName, sizeof((*j)->fileName));
		}
	}
}

void groupsFromFile(std::vector<std::vector<ImageMetaData*>*> &v, char *fileName, std::vector<ImageMetaData> &data) {
	ifstream fin(fileName, ios::in | ios::binary);
	size_t size;
	fin.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	for (size_t i = 0; i < size; i++) {
		vector<ImageMetaData*>* dummy = new vector<ImageMetaData*>();
		size_t numInGroup;
		fin.read(reinterpret_cast<char*>(&numInGroup), sizeof(size_t));
		for (size_t j = 0; j < numInGroup; j++) {
			char fileName[sizeof(ImageMetaData::fileName)];
			fin.read(fileName, sizeof(fileName));
			dummy->push_back(findByName(data, fileName));
		}
		v.push_back(dummy);
	}
}
