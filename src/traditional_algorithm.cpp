#include <time.h>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

vector<Point2i> GrayCenter(Mat src, int direction) {
  vector<Point2i> centerPoint;
  if (direction == 0) {
    for (int i = 0; i < src.rows; i++) {
      int mean = src.at<uchar>(i, 0);
      vector<float> current_value;
      vector<float> current_coordinat;
      int sumvalue = 0;
      int sumcoordinat = 0;
      for (int j = 0; j < src.cols; j++) {
        if (src.at<uchar>(i, j) > mean) {
          mean = src.at<uchar>(i, j);
        }
      }
      if (mean > 0) {
        for (int j = 0; j < src.cols; j++) {
          int current = src.at<uchar>(i, j);
          if (current > (mean - 1)) {
            current_value.push_back(current);
            current_coordinat.push_back(j);
          }
        }
        // 计算灰度重心
        for (int k = 0; k < current_value.size(); k++) {
          sumcoordinat += current_value[k] * current_coordinat[k];
          sumvalue += current_value[k];
        }
        int y = sumcoordinat / sumvalue;
        Point2i inpoint = Point2i(y, i);
        centerPoint.push_back(inpoint);
      }
    }
  } else if (direction == 1) {
    for (int i = 0; i < src.cols; i++) {
      int mean = src.at<uchar>(0, i);
      vector<float> current_value;
      vector<float> current_coordinat;
      int sumvalue = 0;
      int sumcoordinat = 0;
      for (int j = 0; j < src.rows; j++) {
        if (src.at<uchar>(j, i) > mean) {
          mean = src.at<uchar>(j, i);
        }
      }
      if (mean > 0) {
        for (int m = 0; m < src.rows; m++) {
          int current = src.at<uchar>(m, i);
          if (current > (mean - 1)) {
            current_value.push_back(current);
            current_coordinat.push_back(m);
          }
        }
        // 计算灰度重心
        for (int k = 0; k < current_value.size(); k++) {
          sumcoordinat += current_value[k] * current_coordinat[k];
          sumvalue += current_value[k];
        }
        int y = sumcoordinat / sumvalue;
        Point2i inpoint = Point2i(i, y);
        centerPoint.push_back(inpoint);
      }
    }
  }
  return centerPoint;
  vector<Point2i> GrayGravity(Mat src, int direction) {
    Mat grayimg;
    vector<Point2i> centerPoint;
    if (direction == 0) {
      for (int i = 0; i < src.rows; i++) {
        int mean = src.at<uchar>(i, 0);
        vector<float> current_value;
        vector<float> current_coordinat;
        int sumvalue = 0;
        int sumcoordinat = 0;
        for (int j = 0; j < src.cols; j++) {
          if (src.at<uchar>(i, j) > mean) {
            mean = src.at<uchar>(i, j);
          }
        }
        if (mean > 0) {
          for (int j = 0; j < src.cols; j++) {
            int current = src.at<uchar>(i, j);
            if (current > (mean - 1)) {
              current_value.push_back(current);
              current_coordinat.push_back(j);
            }
          }
          // 计算灰度重心
          for (int k = 0; k < current_value.size(); k++) {
            sumcoordinat += current_value[k] * current_coordinat[k];
            sumvalue += current_value[k];
          }
          int y = sumcoordinat / sumvalue;
          Point2i inpoint = Point2i(y, i);
          centerPoint.push_back(inpoint);
        }
      }
    } else if (direction == 1) {
      for (int i = 0; i < src.cols; i++) {
        int mean = src.at<uchar>(0, i);
        vector<float> current_value;
        vector<float> current_coordinat;
        int sumvalue = 0;
        int sumcoordinat = 0;
        for (int j = 0; j < src.rows; j++) {
          if (src.at<uchar>(j, i) > mean) {
            mean = src.at<uchar>(j, i);
          }
        }
        if (mean > 0) {
          for (int m = 0; m < src.rows; m++) {
            int current = src.at<uchar>(m, i);
            if (current > (mean - 1)) {
              current_value.push_back(current);
              current_coordinat.push_back(m);
            }
          }
          // 计算灰度重心
          for (int k = 0; k < current_value.size(); k++) {
            sumcoordinat += current_value[k] * current_coordinat[k];
            sumvalue += current_value[k];
          }
          int y = sumcoordinat / sumvalue;
          Point2i inpoint = Point2i(i, y);
          centerPoint.push_back(inpoint);
        }
      }
    }
    return centerPoint;
  }

  vector<Point2i> StegerLine(Mat src) {
    vector<Point2i> centerPoint;
    // 高斯滤波
    src.convertTo(src, CV_32FC1);
    GaussianBlur(src, src, Size(0, 0), 6, 6);

    // 一阶偏导数
    Mat m1, m2;
    m1 = (Mat_<float>(1, 2) << 1, -1);  // x偏导
    m2 = (Mat_<float>(2, 1) << 1, -1);  // y偏导

    Mat dx, dy;
    filter2D(src, dx, CV_32FC1, m1);
    filter2D(src, dy, CV_32FC1, m2);

    // 二阶偏导数
    Mat m3, m4, m5;
    m3 = (Mat_<float>(1, 3) << 1, -2, 1);      // 二阶x偏导
    m4 = (Mat_<float>(3, 1) << 1, -2, 1);      // 二阶y偏导
    m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);  // 二阶xy偏导

    Mat dxx, dyy, dxy;
    filter2D(src, dxx, CV_32FC1, m3);
    /*cout << dxx.at<float>(0, 0) << endl;
    cout << dxx.at<float>(0, 1) << endl;
    cout << dxx.at<float>(0, 2) << endl;*/
    filter2D(src, dyy, CV_32FC1, m4);
    filter2D(src, dxy, CV_32FC1, m5);

    // hessian矩阵
    double maxD = -1;
    int imgcol = src.cols;
    int imgrow = src.rows;
    vector<double> Pt;
    for (int i = 0; i < imgcol; i++) {
      for (int j = 0; j < imgrow; j++) {
        if (src.at<uchar>(j, i) > 200) {
          Mat hessian(2, 2, CV_32FC1);
          hessian.at<float>(0, 0) = dxx.at<float>(j, i);
          hessian.at<float>(0, 1) = dxy.at<float>(j, i);
          hessian.at<float>(1, 0) = dxy.at<float>(j, i);
          hessian.at<float>(1, 1) = dyy.at<float>(j, i);

          Mat eValue;
          Mat eVectors;
          eigen(hessian, eValue, eVectors);  // 求特征值和特征向量

          double nx, ny;
          double fmaxD = 0;
          if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  // 求特征值最大时对应的特征向量
          {
            nx = eVectors.at<float>(0, 0);
            ny = eVectors.at<float>(0, 1);
            fmaxD = eValue.at<float>(0, 0);
          } else {
            nx = eVectors.at<float>(1, 0);
            ny = eVectors.at<float>(1, 1);
            fmaxD = eValue.at<float>(1, 0);
          }

          double t =
              -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) /
              (nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));

          if (fabs(t * nx) <= 0.5 && fabs(t * ny) <= 0.5) {
            Pt.push_back(i);
            Pt.push_back(j);
          }
        }
      }
    }
    for (int k = 0; k < Pt.size() / 2; k++) {
      Point2i rpt;
      rpt.x = Pt[2 * k + 0];
      rpt.y = Pt[2 * k + 1];
      centerPoint.push_back(rpt);
    }
    return centerPoint;
  }

  vector<Point2i> Zhang_Suen_ThinImg(Mat src) {
    vector<Point2i> centerPoint;
    vector<Point> deleteList;
    int neighbourhood[9];
    int nl = src.rows;
    int nc = src.cols;
    bool inOddIterations = true;
    while (true) {
      for (int j = 1; j < (nl - 1); j++) {
        uchar* data_last = src.ptr<uchar>(j - 1);
        uchar* data = src.ptr<uchar>(j);
        uchar* data_next = src.ptr<uchar>(j + 1);
        for (int i = 1; i < (nc - 1); i++) {
          if (data[i] == 255) {
            int whitePointCount = 0;
            neighbourhood[0] = 1;
            if (data_last[i] == 255)
              neighbourhood[1] = 1;
            else
              neighbourhood[1] = 0;
            if (data_last[i + 1] == 255)
              neighbourhood[2] = 1;
            else
              neighbourhood[2] = 0;
            if (data[i + 1] == 255)
              neighbourhood[3] = 1;
            else
              neighbourhood[3] = 0;
            if (data_next[i + 1] == 255)
              neighbourhood[4] = 1;
            else
              neighbourhood[4] = 0;
            if (data_next[i] == 255)
              neighbourhood[5] = 1;
            else
              neighbourhood[5] = 0;
            if (data_next[i - 1] == 255)
              neighbourhood[6] = 1;
            else
              neighbourhood[6] = 0;
            if (data[i - 1] == 255)
              neighbourhood[7] = 1;
            else
              neighbourhood[7] = 0;
            if (data_last[i - 1] == 255)
              neighbourhood[8] = 1;
            else
              neighbourhood[8] = 0;
            for (int k = 1; k < 9; k++) {
              whitePointCount += neighbourhood[k];
            }
            if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
              int ap = 0;
              if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
              if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
              if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
              if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
              if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
              if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
              if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
              if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
              if (ap == 1) {
                if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0) &&
                    (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
                  deleteList.push_back(Point(i, j));
                } else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0) &&
                           (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
                  deleteList.push_back(Point(i, j));
                }
              }
            }
          }
        }
      }
      if (deleteList.size() == 0) break;
      for (size_t i = 0; i < deleteList.size(); i++) {
        Point tem;
        tem = deleteList[i];
        uchar* data = src.ptr<uchar>(tem.y);
        data[tem.x] = 0;
      }
      deleteList.clear();

      inOddIterations = !inOddIterations;
    }
    for (int i = 0; i < src.rows; i++) {
      for (int j = 0; j < src.cols; j++) {
        if (src.at<uchar>(i, j) == 255) {
          Point2i centerpoint;
          centerpoint.x = j;
          centerpoint.y = i;
          // circle(src, centerpoint, 1, Scalar(0, 0, 255), 1, 8);
          centerPoint.push_back(centerpoint);
        }
      }
    }
    return centerPoint;
  }

  vector<Point2i> DistanceTransform(Mat src) {
    vector<Point2i> centerPoint;
    Mat distImg;
    threshold(src, src, 128, 255, cv::THRESH_BINARY);
    distanceTransform(src, distImg, DIST_L2, DIST_MASK_PRECISE);  // 距离变换细化光条中心
    normalize(distImg, distImg, 0, 1, NORM_MINMAX);
    Point2i maxPoint;
    for (int i = 0; i < distImg.rows; i++) {
      float maxValue = 0;
      for (int j = 0; j < distImg.cols; j++) {
        if (distImg.at<float>(i, j) > maxValue) {
          maxValue = distImg.at<float>(i, j);
          maxPoint.x = j;
          maxPoint.y = i;
        }
      }
      if (maxPoint.x == 0 && maxPoint.y == 0) continue;
      centerPoint.push_back(maxPoint);
    }
    return centerPoint;
  }

  int main(int argc, char** argv) {
    cv::Mat src = cv::imread("./dataset/1.bmp");
    Mat grayImage;
    cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);

    vector<Point2i> centerPoint = StegerLine(grayImage);

    for (const auto& pt : centerPoint) {
      // 在点的位置画一个小圆
      // 参数分别是：目标图像，中心点，半径，颜色，厚度
      cv::circle(src, pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("src", src);
    cv::waitKey(0);

    std::cout << "Testing" << std::endl;

    return 0;
  }