#include<iostream>
#include <opencv2/opencv.hpp>
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/ml.hpp"
// #include "opencv2/objdetect.hpp"
// #include "opencv2/videoio.hpp"
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        // CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}
int main()
{
    // Mat test_image;         test_image = imread("image-digi.png",1);                 // imshow("test_image",test_image);
    Mat test_image;         test_image = imread("image-compare.png",1);                 // imshow("test_image",test_image);
    Mat blur_test_image;    GaussianBlur(test_image, blur_test_image,Size(5,5),0);      // imshow("blur_test", blur_test_image);
    Mat thresh_test_image;  threshold(blur_test_image, thresh_test_image, 190, 255, THRESH_BINARY);     imshow("thresh_test",thresh_test_image);
    Mat canny_test_image;   Canny(thresh_test_image, canny_test_image, 150, 200);       // imshow("canny_test",canny_test_image);
    Mat test_roi_image;
    vector<vector<Point>> test_contour;
    vector<Vec4i> test_hierarchy;
    findContours(canny_test_image, test_contour, test_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    int test_contour_size = test_contour.size();

    cout<<"contour test size: "<<test_contour_size<<endl;
    // waitKey(0);
    int x1,x2,y1,y2,h,w;
    for(int i=0; i<test_contour_size; i++)
    {   
        // int test_contour_size = test_contour.size();
        int contour_Area_i = contourArea(test_contour[i]);
        cout<<"contour Area: "<<contour_Area_i;
        if( contour_Area_i > 150  )
        {
            vector<vector<Point> > contours_poly( test_contour_size );
            approxPolyDP( test_contour[i], contours_poly[i], 3, true );
            vector<Rect> boundRect(test_contour_size );                          // bound = rang buoc
            boundRect[i] = boundingRect( contours_poly[i] );
            x1 = boundRect[i].tl().x;
            y1 = boundRect[i].tl().y;
            x2 = boundRect[i].br().x;
            y2 = boundRect[i].br().y;
            w = x2 - x1;
            h = y2 - y1;
            cout<<h<<"h ";
            if (h>20)
            {
                //ve hinh chu nhat
                rectangle( test_image, boundRect[i].tl(), boundRect[i].br(),Scalar(0,255,0),1);
                Rect rect1(x1,y1,w,h);
                test_roi_image = canny_test_image(rect1);                                       // crop origin image      //crop = cut

                imshow("test_roi_image ", test_roi_image);
                // waitKey(0);
            }
        }

    }                
    imshow("test_image_done", test_image);

    waitKey(0);
    cout<<endl<<"Training complete."<<endl;
}