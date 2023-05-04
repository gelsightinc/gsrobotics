// File Name: tracking.cpp
// Author: Shaoxiong Wang
// Create Time: 2018/12/20 20:35

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <time.h>
#include "tracking_class.h"

using namespace cv;
using namespace std;

#define CAMERA_WIDTH 720
#define CAMERA_HEIGHT 576

#define SCALE_DOWN 1

Point_t centers[MAXNM];
Matching matcher;

void example(Mat gray){

    for(int y=0;y<gray.rows;y++)
    {
        for(int x=0;x<gray.cols;x++)
        {
            // get pixel
            Vec3b color = gray.at<Vec3b>(Point(x,y));
            
            // cout<<color<<" ";

            // set pixel
            // image.at<Vec3b>(Point(x,y)) = color;
        }
        cout<<endl;
    }
}

Mat find_marker(Mat frame) {
    Mat blur, I, gray, mask, mask1, mask2;
    Mat planes[3];

    // int threshold_rg = 40;
    // int threshold_gray = 20;
    // int threshold_gray = 15;
    int threshold_gray = 15;

    GaussianBlur(frame, blur, Size( 21 * 4 + 1, 21 * 4 + 1), 0, 0);
    // GaussianBlur(frame, blur, Size( 25, 25), 0, 0);
    // I = frame - blur;
    I = blur * 1.0 - frame;
    // mask = I;

    // cv::cvtColor(I, gray, CV_BGR2GRAY);

    // mask = gray > threshold_gray;

    split(I, planes);
    mask1 = (planes[0] > threshold_gray) | (planes[1] > threshold_gray) | (planes[2] > threshold_gray);
    mask2 = (planes[0] > threshold_gray / 3) & (planes[1] > threshold_gray / 3) & (planes[2] > threshold_gray / 3);
    // cout<<mask;
    mask = mask1 & mask2;
    return mask;
}

void MyFilledCircle( Mat img, Point center )
{
 int thickness = 2;
 // int thickness = -1;
 int lineType = 8;

 circle( img,
         center,
         10,
         Scalar( 0, 0, 255 ),
         thickness,
         lineType );
}

void marker_center(Mat mask, Mat frame) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // cout<<contours.size()<<" "<<contours[0];

    for(int y=0;y<mask.rows;y++)
    {
        for(int x=0;x<mask.cols;x++)
        {
            // get pixel
            Vec3b color = frame.at<Vec3b>(Point(x,y));
            Vec3b color_mask = mask.at<Vec3b>(Point(x,y));
            // Vec3b color(0, 0, 0);
            // color[0] = color[0] - color_mask[0] * 0.2;
            
            // cout<<color<<" ";

            // set pixel
            frame.at<Vec3b>(Point(x,y)) = color;
        }
        // cout<<endl;
    }

    int marker_size_min = 30 * SCALE_DOWN * SCALE_DOWN, marker_size_max = 360 * SCALE_DOWN * SCALE_DOWN;
    // int marker_size_min = 10, marker_size_max = 80;
    // int marker_size_min = 10, marker_size_max = 80;
    int count = 0;

    /// Get the moments
    vector<Moments> mu(contours.size() );

    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );

    for( int i = 0; i< contours.size(); i++) {
        int a = contourArea(contours[i]);
        // cout<<endl<<a;
        if (a < marker_size_min || a > marker_size_max) continue;

        // cout<<"+++";

        mu[i] = moments( contours[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );

        // cout << mc[i].x <<" " << mc[i].y << endl;
        // MyFilledCircle( frame, Point( mc[i].x, mc[i].y) );
        centers[count].x = mc[i].x;
        centers[count].y = mc[i].y;
        centers[count].id = count;

        count++;

    }
    cout<<"COUNT "<<count<<"\t";

    clock_t t;
    t = clock();

    matcher.init(centers, count);
    matcher.run();


    int i, j;
    for(i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            Point a(matcher.O[i][j].x, matcher.O[i][j].y), b(matcher.MinD[i][j].x + 2 * (matcher.MinD[i][j].x - matcher.O[i][j].x), matcher.MinD[i][j].y + 2 * (matcher.MinD[i][j].y - matcher.O[i][j].y));
            Vec3b color(0,0,255);
            if (matcher.MinOccupied[i][j] <= -1) {
                printf("%d %d INFERED!!\n",i, j);
                color[0] = 127;
                color[1] = 127;
                color[2] = 255;
            }
            // else{
            // }

            arrowedLine(frame, a, b, color, 4);
        }
    }

    printf("track: %.6lf seconds\n", ((double)clock() - t)/CLOCKS_PER_SEC);
}

Mat init(Mat img) {
    int offset_x0 = 0, offset_x1 = 0;
    int offset_y0 = 0, offset_y1 = 0;

    cv::Rect roi;
    roi.x = offset_x0;
    roi.y = offset_y0;
    roi.width = img.size().width - (offset_x0 + offset_x1);
    roi.height = img.size().height - (offset_y0 + offset_y1);

    return img(roi);
}

Mat init_HSR(Mat img) {
    float WARP_W = 290, WARP_H = 290;
    Point2f src[4] = {
        Point2f(185, 260),
        Point2f(450, 260),
        Point2f(0, 460),
        Point2f(620, 460)
    };
    Point2f dst[4] = {
        Point2f(0, 0),
        Point2f(WARP_W, 0),
        Point2f(0, WARP_H),
        Point2f(WARP_W, WARP_H)
    };
    Mat matrix = getPerspectiveTransform(src, dst);
    Mat ret;
    warpPerspective(img, ret, matrix, Size(WARP_W, WARP_H));
    return ret;
}

int main(int argc, char** argv){
    // cout<< dist_sqr(O[0][0], O[0][1]);

    // VideoCapture capture("../data/GelSight_Shear_Test_2.mov");
    // VideoCapture capture("../data/GelSight_Twist_Test_2.mov");
    VideoCapture capture(1);

    VideoWriter output("HSR.avi",CV_FOURCC('M','J','P','G'),30, Size(290,290));

    // VideoCapture capture(0);
    Mat frame, mask;
    int key;

    namedWindow("frame", WINDOW_OPENGL); 

    clock_t t;
    t = clock();
    int cnt_frame = 0;
    while(1){
        cnt_frame++;

        t = clock();

        cout<<"frame "<<cnt_frame<<"\t";
        capture >> frame;   
        if (cnt_frame < 25) continue;

        // if (cnt_frame > 1 && cnt_frame < 485) continue;
        // if (cnt_frame > 380) break;
        if(frame.empty()) break;

        frame = init_HSR(frame);
        cv::resize(frame, frame, cv::Size(), SCALE_DOWN, SCALE_DOWN);

        printf("loading: %.6lf seconds\t", ((double)clock() - t)/CLOCKS_PER_SEC);
        
        t = clock();

        mask = find_marker(frame);

        printf("centers: %.6lf seconds\t", ((double)clock() - t)/CLOCKS_PER_SEC);
        

        t = clock();

        marker_center(mask, frame);

        printf("matching: %.6lf seconds\n", ((double)clock() - t)/CLOCKS_PER_SEC);
        

        cout << frame.size;
        cout<<endl;
        // cv::resize(frame, frame, cv::Size(), 0.5, 0.5);;
        imshow("frame", frame);
        // imshow("frame", frame);
        output.write(frame);
        key = waitKey(1);
        // break;

        if (key == 'q')
            break;

    }
    capture.release();
    output.release();
    destroyAllWindows();
    return 0;
}