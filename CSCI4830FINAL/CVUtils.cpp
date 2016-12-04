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
#define MAX_NPIX 125000.0
#define INLIER_THRESH 2.5f

#define HIST_THRESH 0.3
#define KEYPOINT_THRESH 0.040
#define HOMOGRAPHY_THRESH 0.020
#define MINIMUM_KEYPOINTS 8

#define HOMOGRAPHY_SCALE_THRESH_LOW 0.8
#define HOMOGRAPHY_SCALE_THRESH_HIGH 1.2
#define HOMOGRAPHY_THETA_THRESH_LOW 30.0
#define	HOMOGRAPHY_THETA_THRESH_HIGH 60.0

#define MINIMUM_SPREAD 0.1

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

bool* precomputeHistogramMatches(vector<ImageMetaData> &v) {
	bool* table = reinterpret_cast<bool*>(malloc(v.size()*v.size()*sizeof(bool)));
	for (vector<ImageMetaData>::iterator i = v.begin(); i < v.end(); i++) {
		for (vector<ImageMetaData>::iterator j = v.begin(); j < v.end(); j++) {
			Mat img1, img2;
			cvtColor(imread((*i).fileName), img1, COLOR_BGR2HSV);
			cvtColor(imread((*j).fileName), img2, COLOR_BGR2HSV);
			MatND hist_img1;
			MatND hist_img2;
			int channels[] = { 0,1 };
			int histSize[] = { 50,60 };
			float h_ranges[] = { 0, 180 };
			float s_ranges[] = { 0, 256 };
			const float* ranges[] = { h_ranges, s_ranges };
			calcHist(&img1, 1, channels, Mat(), hist_img1, 2, histSize, ranges, true, false);
			calcHist(&img2, 1, channels, Mat(), hist_img2, 2, histSize, ranges, true, false);
			double comparisonValue = compareHist(hist_img1, hist_img2, CV_COMP_CORREL);
			table[(i-v.begin())*v.size() + (j-v.begin())] = (comparisonValue < HIST_THRESH);
		}
	}
	return table;
}

bool goodHomography(Mat &H, vector<KeyPoint> &kp1) {
	/*double a = H.at<double>(0, 0);
	double b = H.at<double>(0, 1);
	double c = H.at<double>(0, 2);
	double d = H.at<double>(0, 0);
	double e = H.at<double>(0, 1);
	double f = H.at<double>(0, 2);

	double scalex = sqrt(a*a + b*b);
	double scaley = (a*e - b*d) / (scalex);
	double shear = (a*d + b*e) / (a*e - b*d);
	double theta = atan2(b, a);

	bool success = true;
	success = success && (scalex > HOMOGRAPHY_SCALE_THRESH_LOW) && (scalex < HOMOGRAPHY_SCALE_THRESH_HIGH);
	success = success && (scaley > HOMOGRAPHY_SCALE_THRESH_LOW) && (scaley < HOMOGRAPHY_SCALE_THRESH_HIGH);
	success = success && (theta > -HOMOGRAPHY_THETA_THRESH) && (theta < HOMOGRAPHY_THETA_THRESH);
	return success;*/

	float meanx = 0, meany = 0, count = 0;
	for (vector<KeyPoint>::iterator i = kp1.begin(); i < kp1.end(); i++) {
		meanx += (*i).pt.x;
		meany += (*i).pt.y;
		count += 1;
	}
	meanx /= count;
	meany /= count;

	vector<Point2f> inputPoints(2);
	inputPoints[0] = Point2f(meanx, meany);
	inputPoints[1] = Point2f(meanx+10, meany+10);
	vector<Point2f> outputPoints(2);
	perspectiveTransform(inputPoints, outputPoints, H);
	float dx, dy;
	dx = outputPoints[1].x - outputPoints[0].x;
	dy = outputPoints[1].y - outputPoints[0].y;
	float scale, angle;
	scale = sqrt(dx*dx + dy*dy)/sqrt(200);
	angle = 180.0/3.1415*atan2(dy, dx);
	if (angle > 180) angle -= 360;
	if (angle < -180) angle += 360;
	bool success = true;
	success = success && (scale > HOMOGRAPHY_SCALE_THRESH_LOW) && (scale < HOMOGRAPHY_SCALE_THRESH_HIGH);
	success = success && (angle > HOMOGRAPHY_THETA_THRESH_LOW) && (angle < HOMOGRAPHY_THETA_THRESH_HIGH);
	return success;
}

bool duplicateDetect(ImageMetaData &imd1, ImageMetaData &imd2) {

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
			kp1.push_back(imd1.keypoints[first.queryIdx]);
			kp2.push_back(imd2.keypoints[first.trainIdx]);
			pt1.push_back(imd1.keypoints[first.queryIdx].pt);
			pt2.push_back(imd2.keypoints[first.trainIdx].pt);
		}
	}
	//FlannBasedMatcher matcher;
	//std::vector<DMatch> matches;
	//vector<KeyPoint> kp1, kp2, inliers1, inliers2;
	//vector<Point2f> pt1, pt2;
	//vector<DMatch> good_matches;
	//Mat descriptors1, descriptors2;
	//imd1.descriptors.convertTo(descriptors1, CV_32F);
	//imd2.descriptors.convertTo(descriptors2, CV_32F);
	//matcher.match(descriptors1, descriptors2, matches);
	//double minDist = INFINITY, maxDist = 0;
	//for (size_t i = 0; i < matches.size(); i++) {
	//	double dist = matches[i].distance;
	//	minDist = min(dist, minDist);
	//	maxDist = max(dist, maxDist);
	//}

	//for (size_t i = 0; i < matches.size(); i++) {
	//	if (matches[i].distance < 2 * minDist) {
	//		kp1.push_back(imd1.keypoints[matches[i].queryIdx]);
	//		kp2.push_back(imd2.keypoints[matches[i].trainIdx]);
	//		pt1.push_back(imd1.keypoints[matches[i].queryIdx].pt);
	//		pt2.push_back(imd2.keypoints[matches[i].trainIdx].pt);
	//	}
	//}
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

	if (inliers1.size() < MINIMUM_KEYPOINTS) return false;

	float keypointRatio = static_cast<float>(kp1.size() * 2) / (imd1.keypoints.size() + imd2.keypoints.size());
	if(keypointRatio < KEYPOINT_THRESH) return false;
	//printf("Successful keypoint matches: %.2f%%\n", keypointRatio*100);
	float homographyRatio = static_cast<float>(inliers1.size() * 2) / (imd1.keypoints.size() + imd2.keypoints.size());
	if(homographyRatio < HOMOGRAPHY_THRESH) return false;
	//printf("Successful homography matches: %.2f%%\n", homographyRatio * 100);
	//successful = successful && (goodHomography(H, inliers1));
	//float meanx1=0, meany1 = 0, meanx2 = 0, meany2 = 0, stdx1 = 0, stdx2 = 0, stdy1 = 0, stdy2 = 0;
	//size_t i;
	//for (i = 0; i < inliers1.size(); i++) {
	//	meanx1 += inliers1[i].pt.x;
	//	meany1 += inliers1[i].pt.y;
	//	meanx2 += inliers2[i].pt.x;
	//	meany2 += inliers2[i].pt.y;
	//}
	//meanx1 /= i;
	//meany1 /= i;
	//meanx2 /= i;
	//meany2 /= i;

	//for (i = 0; i < inliers1.size(); i++) {
	//	stdx1 += pow(inliers1[i].pt.x - meanx1, 2);
	//	stdy1 += pow(inliers1[i].pt.y - meany1, 2);
	//	stdx2 += pow(inliers2[i].pt.x - meanx2, 2);
	//	stdy2 += pow(inliers2[i].pt.y - meany2, 2);
	//}
	//stdx1 /= i;
	//stdx1 = sqrt(stdx1);
	//stdy1 /= i;
	//stdy1 = sqrt(stdy1);
	//stdx2 /= i;
	//stdx2 = sqrt(stdx2);
	//stdy2 /= i;
	//stdy2 = sqrt(stdy2);

	//if (stdx1 / imd1.usedSize.width < MINIMUM_SPREAD) return false;
	//if (stdx2 / imd1.usedSize.width < MINIMUM_SPREAD) return false;
	//if (stdy1 / imd1.usedSize.height < MINIMUM_SPREAD) return false;
	//if (stdy2 / imd1.usedSize.height < MINIMUM_SPREAD) return false;

	if (!goodHomography(H, inliers1)) return false;

	Mat outImage;
	Mat image1, image2;
	resize(imread(imd1.fileName), image1, imd1.usedSize);
	resize(imread(imd2.fileName), image2, imd2.usedSize);
	drawMatches(image1, inliers1, image2, inliers2, good_matches, outImage);

	destroyWindow("Homography points");
	namedWindow("Homography points", WINDOW_AUTOSIZE);
	imshow("Homography points", outImage);
	waitKey(10);

	return true;
}