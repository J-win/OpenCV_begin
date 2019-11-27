#include "opencv2/opencv.hpp"
using namespace cv;

int main() {
	Mat image, gray, def, canny, canny_harris, harris, norm, dist;
	image = imread("Sample.png", CV_LOAD_IMAGE_COLOR);
	imshow("DisplayPicture1", image);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("DisplayPicture2", gray);
	int max = 0, min = 255;
	for (int i = 0; i < gray.rows; ++i) {
		for (int j = 0; j < gray.cols; ++j) {
			int inten = gray.at<uchar>(i,j);
			if (max < inten)
				max = inten;
			if (min > inten)
				min = inten;
		}
	}
	gray.convertTo(def, CV_32F, 1.0 / 255.0);
	int k = 255 / (max - min);
	int b = -(min * 255) / (max - min);
	for (int i = 0; i < def.rows; ++i) {
		for (int j = 0; j < def.cols; ++j) {
			def.at<float>(i, j) = def.at<float>(i, j) * k + b;
		}
	}
	imshow("DisplayPicture3", def);
	def.convertTo(canny, CV_8U, 255.0);
	Canny(canny, canny, 50, 150, 3);
	imshow("DisplayPicture4", canny);

	canny.copyTo(canny_harris);
	harris = Mat::zeros(canny.size(), CV_32FC1);

	cornerHarris(canny, harris, 5, 3, 0.04);
	imshow("DisplayPicture5", harris);
	normalize(harris, norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	int t = 90;
	for (int i = 0; i < norm.rows; i++) {
		for (int j = 0; j < norm.cols; j++) {
			if ((int)norm.at<float>(i, j) > t) {
				circle(canny_harris, Point(j, i), 2, Scalar(255, 255, 255), 2, 8, 0);
			}
		}
	}

	bitwise_not(canny_harris, canny_harris);

	distanceTransform(canny_harris, dist, DIST_L1, 3);
	imshow("DisplayPicture6", canny_harris);
	imshow("DisplayPicture7", dist);

	Mat blur = Mat::zeros(def.size(), CV_32FC1);
	Mat integ = Mat::zeros(def.size(), CV_32FC1);
	integral(def, integ, CV_32FC1);

	for (int i = 0; i < def.rows; i++) {
		for (int j = 0; j < def.cols; j++) {
			int d = dist.at<float>(i, j);
			if (d % 2 == 0) {
				d++;
			}
			int x = d;
			int y = d;
			float sum = 0;
			if (d != 0) {
				if (i - y < 0)
					y = i;
				if (i + y >= integ.rows)
					y = int(integ.rows - i - 1);
				if (j - x < 0)
					x = j;
				if (j + x >= integ.cols)
					x = int(integ.cols - j - 1);

				sum = integ.at<float>(i + y, j + x) - integ.at<float>(i - y, j + x) - integ.at<float>(i + y, j - x) + integ.at<float>(i - y, j - x);
				sum /= 4 * x * y;
				blur.at<float>(i, j) = sum;
			}
			else {
				blur.at<float>(i, j) = def.at<float>(i, j);
			}
		}
	}

	blur.convertTo(blur, CV_8U, 255.0);
	imshow("DisplayPicture8", blur);

	waitKey();
	system("pause");
	return 0;
}