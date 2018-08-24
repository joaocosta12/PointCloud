

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;


int main()
{
    Mat img1, img2;
    img1 = imread("left2.png");//, CV_LOAD_IMAGE_GRAYSCALE
    img2 = imread("right2.png");

  

    double cm1[3][3] = {{9.597910e+02, 0.000000e+00, 6.960217e+02}, {0.000000e+00, 9.569251e+02, 2.241806e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double cm2[3][3] = {{9.037596e+02, 0.000000e+00, 6.957519e+02}, {0.000000e+00, 9.019653e+02, 2.242509e+02}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double d1[1][5] = {{-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02}};
    double d2[1][5] = {{-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02}};

    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);

    cout << "Calibration matrix left:\n" << CM1 << endl;
    cout << "Distorstion matrix left:\n" << D1 << endl;
    cout << "Calibration matrix right:\n" << CM2 << endl;
    cout << "Distorstion matrix right:\n" << D2 << endl;


    double r[3][3] = {{9.995599e-01, 1.699522e-02, -2.431313e-02},{-1.704422e-02, 9.998531e-01, -1.809756e-03},{2.427880e-02, 2.223358e-03, 9.997028e-01}};
    double t[3][1] = {{-4.731050e-01}, {5.551470e-03}, {-5.250882e-03}};


    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);


    //Mat   R, T;
    Mat R1, R2, T1, T2, Q, P1, P2;

    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

    cout << "R2:\n" << R2 << endl;
    cout << "P2:\n" << P2 << endl;

    double rr1[3][3] = {{9.998817e-01, 1.511453e-02, -2.841595e-03},{-1.511724e-02, 9.998853e-01, -9.338510e-04},{2.827154e-03, 9.766976e-04, 9.999955e-01}};
    double rr2[3][3] = {{9.998321e-01, -7.193136e-03, 1.685599e-02},{7.232804e-03, 9.999712e-01, -2.293585e-03},{-1.683901e-02, 2.415116e-03, 9.998553e-01}};
    double pp1[3][4] = {{7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01},{0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01},{0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03}};
    double pp2[3][4] = {{7.215377e+02, 0.000000e+00, 6.095593e+02, -3.395242e+02},{0.000000e+00, 7.215377e+02, 1.728540e+02, 2.199936e+00},{0.000000e+00, 0.000000e+00, 1.000000e+00, 2.729905e-03}};
   

    Mat RR1 (3,3, CV_64FC1, rr1);
    Mat RR2 (3,3, CV_64FC1, rr2);
    Mat PP1 (3,4, CV_64FC1, pp1);
    Mat PP2 (3,4, CV_64FC1, pp2);

    Mat map11, map12, map21, map22;
    Size img_size = img1.size();
    initUndistortRectifyMap(CM1, D1, RR1, PP1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, RR2, PP2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    //remap(img1, img1r, map11, map12, INTER_LINEAR);
    //  remap(img2, img2r, map21, map22, INTER_LINEAR);
    //img1 = img1r;
    //img2 = img2r;

    int sadSize = 3;
    StereoSGBM sbm;
    sbm.SADWindowSize = sadSize;
    sbm.numberOfDisparities = 144;//144; 128
    sbm.preFilterCap = 10; //63
    sbm.minDisparity = 0; //-39; 0
    sbm.uniquenessRatio = 10;
    sbm.speckleWindowSize = 100;
    sbm.speckleRange = 32;
    sbm.disp12MaxDiff = 1;
    sbm.fullDP = true;
    sbm.P1 = sadSize*sadSize*4;
    sbm.P2 = sadSize*sadSize*32;

    Mat disp, disp8;
    sbm(img1, img2, disp);

    //disp = imread("disp.png", CV_LOAD_IMAGE_GRAYSCALE);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);


    Mat points, points1;
    reprojectImageTo3D(disp, points, Q, true);
    cvtColor(points, points1, CV_BGR2GRAY);

    Mat img3;
    img3 = imread("semantica.png");
    
	ofstream semantica_file;
    semantica_file.open ("semantica.xyz");
    for(int i = 0; i < img3.rows; i++) {
        for(int j = 0; j < img3.cols; j++) {
                semantica_file << static_cast<unsigned>(img3.at<Vec3b>(i,j)[0]) << " " << static_cast<unsigned>(img3.at<Vec3b>(i,j)[1]) << " " << static_cast<unsigned>(img3.at<Vec3b>(i,j)[2])  << endl;
        }
    }
    semantica_file.close();
    
    ofstream point_cloud_file;
    point_cloud_file.open ("point_cloud.xyz");
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            if(points.at<Vec3f>(i,j)[2] < 10) {
                point_cloud_file << points.at<Vec3f>(i,j)[0] << " " << points.at<Vec3f>(i,j)[1] << " " << points.at<Vec3f>(i,j)[2] << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << endl; 
            }
        }
    }
    point_cloud_file.close();


    ofstream color_cloud_file;
    color_cloud_file.open ("color_cloud.xyz");
    for(int i = 0; i < points.rows; i++) {
        for(int j = 0; j < points.cols; j++) {
            if(points.at<Vec3f>(i,j)[2] < 10) {
                color_cloud_file << points.at<Vec3f>(i,j)[0] << " " << points.at<Vec3f>(i,j)[1] << " " << points.at<Vec3f>(i,j)[2] << " " << static_cast<unsigned>(img1.at<uchar>(i,j)) << " " << static_cast<unsigned>(img3.at<Vec3b>(i,j)[0]) << " " << static_cast<unsigned>(img3.at<Vec3b>(i,j)[1]) << " " << static_cast<unsigned>(img3.at<Vec3b>(i,j)[2])  << endl;
            }
        }
    }
    color_cloud_file.close();


    imshow("Img1", img1);
    imshow("Img2", img2);
    imshow("Img3", img3);
    imshow("points", points);
    imshow("points1", points1);


    waitKey(0);

    return 0;
}

