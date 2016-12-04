#include "CVUtils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <set>

using namespace cv;
using namespace std;

//MUST BE FLOAT!
#define MAX_NPIX 250000.0
#define INLIER_THRESH 2.5f

#define HIST_THRESH 0.3f
#define KEYPOINT_THRESH 0.040
#define HOMOGRAPHY_THRESH 0.020

ImageMetaData* findByName(std::vector<ImageMetaData> &v, const char *fileName) {
	for (vector<ImageMetaData>::iterator i = v.begin(); i < v.end(); i++) {
		if (strcmp(i->fileName, fileName) == 0) {
			return &*i;
		}
	}
	return NULL;
}

ImageMetaData computeDescriptors(char* fileName) {
	Mat image = imread(fileName, IMREAD_GRAYSCALE);
	size_t nPix = image.cols * image.rows;
	Size usedSize;

	//If the image is bad, make junk metadata and print an error
	if (image.empty()) {
		printf("Bad filename: %s\n", fileName);
		ImageMetaData imd;
		strcpy(imd.fileName, fileName);
		imd.keypoints = std::vector<KeyPoint>();
		imd.descriptors = Mat();
		imd.usedSize = image.size();
		return imd;
	}

	//If the image is too big, scale it down
	if (nPix > MAX_NPIX) {
		Mat dst;
		Size newSize(static_cast<int>(image.cols * sqrt(MAX_NPIX / nPix)),
			static_cast<int>(image.rows * sqrt(MAX_NPIX / nPix)));
		resize(image, dst, newSize);
		image = dst;
		usedSize = newSize;
	}
	else {
		usedSize = Size(image.cols, image.rows);
	}

	//Find the keypoints and descriptors
	Ptr<AKAZE> detector = AKAZE::create();
	std::vector<KeyPoint> keypoints;
	Mat descriptors;
	detector->detectAndCompute(image, noArray(), keypoints, descriptors);

	//return calculated values
	ImageMetaData imd;
	strcpy(imd.fileName, fileName);
	imd.keypoints = keypoints;
	imd.descriptors = descriptors;
	imd.usedSize = usedSize;
	return imd;
}

bool duplicateDetect(ImageMetaData &imd1, ImageMetaData &imd2) {
	//-----------------QUICK PASS HISTOGRAM COMPARE------------------
	Mat img1, img2;
	cvtColor(imread(imd1.fileName), img1, COLOR_BGR2HSV);
	cvtColor(imread(imd2.fileName), img2, COLOR_BGR2HSV);
	MatND hist_img1;
	MatND hist_img2;
	int channels[] = { 0,1 };
	int histSize[] = { 50,60 };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	calcHist(&img1, 1, channels, Mat(), hist_img1, 2, histSize, ranges, true, false);
	calcHist(&img2, 1, channels, Mat(), hist_img2, 2, histSize, ranges, true, false);
	float comparisonValue = compareHist(hist_img1, hist_img2, CV_COMP_CORREL);
	if (comparisonValue < HIST_THRESH) return false;

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(imd1.descriptors, imd2.descriptors, nn_matches, 2);

	vector<KeyPoint> kp1, kp2, inliers1, inliers2;
	vector<Point2f> pt1, pt2;
	vector<DMatch> good_matches;

	//set<int> used_indices;

	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < 0.8f * dist2 /*&& used_indices.find(first.trainIdx) == used_indices.end()*/) {
			int new_i = static_cast<int>(kp1.size());
			kp1.push_back(imd1.keypoints[first.queryIdx]);
			kp2.push_back(imd2.keypoints[first.trainIdx]);
			pt1.push_back(imd1.keypoints[first.queryIdx].pt);
			pt2.push_back(imd2.keypoints[first.trainIdx].pt);
			//used_indices.insert(first.trainIdx);
		}
	}
	if (pt1.size() == 0) return false;

	Mat H = findHomography(pt1, pt2, CV_RANSAC);
	if (H.empty()) return false;
	for (int i = 0; i < kp1.size(); i++) {
		Mat col = Mat::ones(3, 1, CV_64F);
		col.at<double>(0) = kp1[i].pt.x;
		col.at<double>(1) = kp1[i].pt.y;

		col = H * col;
		col /= col.at<double>(2);
		float dist = static_cast<float>(sqrt(pow(col.at<double>(0) - kp2[i].pt.x, 2) +
			pow(col.at<double>(1) - kp2[i].pt.y, 2)));

		if (dist < INLIER_THRESH) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(kp1[i]);
			inliers2.push_back(kp2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	//vector<Point2f> box(4);
	//box[0] = cvPoint(0, 0);
	//box[1] = cvPoint(image1.cols, 0);
	//box[2] = cvPoint(image1.cols, image1.rows);
	//box[3] = cvPoint(0, image1.rows);
	//vector<Point2f> map(4);
	//perspectiveTransform(box, map, H);
	//map[0] += Point2f(image1.cols, 0);
	//map[1] += Point2f(image1.cols, 0);
	//map[2] += Point2f(image1.cols, 0);
	//map[3] += Point2f(image1.cols, 0);

	//line(outImage, map[0], map[1], Scalar(0, 255, 0));
	//line(outImage, map[1], map[2], Scalar(0, 255, 0));
	//line(outImage, map[2], map[3], Scalar(0, 255, 0));
	//line(outImage, map[3], map[0], Scalar(0, 255, 0));

	bool successful = true;

	float keypointRatio = static_cast<float>(kp1.size() * 2) / (imd1.keypoints.size() + imd2.keypoints.size());
	successful = successful && (keypointRatio > KEYPOINT_THRESH);
	//printf("Successful keypoint matches: %.2f%%\n", keypointRatio*100);
	float homographyRatio = static_cast<float>(inliers1.size() * 2) / (imd1.keypoints.size() + imd2.keypoints.size());
	successful = successful && (homographyRatio > HOMOGRAPHY_THRESH);
	//printf("Successful homography matches: %.2f%%\n", homographyRatio * 100);

	if (successful) {
		Mat outImage;
		Mat image1, image2;
		resize(imread(imd1.fileName), image1, imd1.usedSize);
		resize(imread(imd2.fileName), image2, imd2.usedSize);
		drawMatches(image1, inliers1, image2, inliers2, good_matches, outImage);

		namedWindow("Homography points", WINDOW_AUTOSIZE);
		imshow("Homography points", outImage);
		waitKey(1);
	}
	return successful;
}