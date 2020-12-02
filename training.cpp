// source           https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python?fbclid=IwAR3e5Tc_VazcNCmlUsAyG1AnTEwmct0qu_wnruqKvr4lH8H7lnHGdljVjV4
// Knearest         https://answers.opencv.org/question/53022/how-to-use-knearest-and-ann-in-opencv-30/?fbclid=IwAR2Nq99S5_xwDZPOIMGLDj5dTMNVxSCxTKphFn7zTMW4PUjmtdapl-qHPtU
// convert_to_m     https://docs.opencv.org/3.4/d0/df8/samples_2cpp_2train_HOG_8cpp-example.html
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv::ml;
using namespace cv;
#define size_image 30
// #define file_name_image_test    "1.JPG"
// #define file_name_image_test    "2.JPG"
#define file_name_image_test    "image-test.JPG"

#define git_clone abcd

void convert_to_1d(Mat_<float> Mat_2d, Mat_<float> Mat_out)
{
    for(int i=0; i<size_image; i++)
    {
        Mat_2d.row(i).copyTo(Mat_out(Rect(size_image*i, 0, size_image, 1)));
    }
}

int main( ) {
    ofstream mySampleFile, myTestFile, myTrainingFile; 
    mySampleFile.open("file-text/sample.txt", std::ofstream::out | std::ofstream::trunc);           // clear all data of file
    myTrainingFile.open("file-text/training.txt", std::ofstream::out | std::ofstream::trunc);       // clear all data of file
    myTestFile.open("file-text/test.txt", std::ofstream::out | std::ofstream::trunc);               // clear all data of file

    // -------read all image--------
    vector<String> file_name;
    Mat_<float> image_table=Mat(0,0,CV_8UC1);
    Mat_<float> image_readed;
    vector<int> label_vector;
    int num_image, stt_image=0;
    String text;

    for(int label=9; label>=0; label--)
    {
        text = format("image-data/%d/*.jpg", label);
        glob(text, file_name, false);
        num_image = file_name.size();
        for (int j=0; j<num_image; j++)
        {
            stt_image++;
            image_readed = imread(file_name[j],0);
            resize(image_readed, image_readed, Size(size_image,size_image),0,0);
            image_table.push_back(image_readed);
            label_vector.push_back(label);
            cout<<stt_image<<" image readed"<<endl;
        }
    }
    num_image = stt_image;

    //--------training data: image_table, label_vector---------
    Mat_<float> train_data=Mat::zeros(num_image, size_image*size_image, CV_8UC1);
    Mat_<int> label_Mat = Mat::zeros(1, num_image, CV_8UC1);
    Ptr<KNearest> knear(KNearest::create());

    for(int i=0; i< size_image*num_image; i++){                              // convert matrix 2d to matrix 1d
        image_table.row(i).copyTo(train_data(cvRect(size_image*(i%size_image),i/size_image,size_image,1)));
    }
    memcpy(label_Mat.data, label_vector.data(), label_vector.size()*sizeof(int));   //convert vector<int> to Mat_<int>
    knear->train(train_data, ROW_SAMPLE, label_Mat);
    cout<<"train data size: "<<train_data.size()<<endl;
    
    mySampleFile    <<label_Mat     <<endl;
    myTrainingFile  <<train_data    <<endl;
    myTrainingFile  <<image_table        <<endl;
    cout<<"train_data: DONE"<<endl;

    //------------testing part-------------
    //read and crop image
    Mat image_test_origin;
    Mat image_test_gray;
    Mat image_test_tranf;
    image_test_origin = imread(file_name_image_test,1);
    cvtColor(image_test_origin, image_test_gray, COLOR_BGR2GRAY);
    GaussianBlur(image_test_gray, image_test_tranf,Size(5,5),0);
    threshold(image_test_tranf, image_test_tranf, 190, 255, THRESH_BINARY);
    Canny(image_test_tranf, image_test_tranf, 150, 200,5);

    vector<Mat> test_image_vector;
    vector<vector<Point>> test_contour;
    vector<Vec4i> test_hierarchy;
    findContours(image_test_tranf, test_contour, test_hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat test_roi_image, test_roismall_image, resu, dist, test_roismall_image_conveted;
    Mat_<int> resu_mat=Mat::zeros(1,size_image*size_image,CV_8UC1);
    Mat_<int> dist_mat=Mat::zeros(1,size_image*size_image,CV_8UC1);
    vector<Mat> resu_vector;
    Mat1f test_float_image_1d=Mat::zeros(1, size_image*size_image,CV_8UC1);

    int test_contour_size = test_contour.size();
    int temp_test=1, K=1;
    int x1, x2, y1, y2, w, h, key;
    cout<<"\ncontour test size: "<<test_contour_size<<endl;
    for(int i2=0; i2<test_contour_size; i2++)
    {   
        int contour_Area_i = contourArea(test_contour[i2]);
        if( contour_Area_i > 150  )
        {
            vector<vector<Point> > contours_poly( test_contour_size );
            approxPolyDP( test_contour[i2], contours_poly[i2], 3, true );
            vector<Rect> boundRect(test_contour_size );                          // bound = rang buoc
            boundRect[i2] = boundingRect( contours_poly[i2] );
            x1 = boundRect[i2].tl().x;
            y1 = boundRect[i2].tl().y;
            x2 = boundRect[i2].br().x;
            y2 = boundRect[i2].br().y;
            w = x2 - x1;
            h = y2 - y1;
            if (h>20)
            {
                //draw rectangle around contours
                rectangle( image_test_origin, Point(x1,y1), Point(x2,y2), Scalar(0,255,0), 2);
                Rect rect_test(x1+1,y1+1,w-1,h-1);
                test_roi_image = image_test_gray(rect_test);                          // crop origin image
                resize(test_roi_image, test_roismall_image, Size(size_image,size_image), 0, 0);
                Mat1f test_float_image;
                test_float_image.push_back(test_roismall_image);
                convert_to_1d(test_float_image, test_float_image_1d);

                test_image_vector.clear();
                test_image_vector.push_back(test_roismall_image);
                myTestFile      <<test_float_image_1d<<endl;
                myTestFile      <<test_float_image<<endl;

                knear->findNearest(test_float_image_1d, K, noArray(), resu, dist );     // find label of image
                mySampleFile    <<resu;
                cout<<"result:"<<resu.row(0)<<"  distance:"<<dist<<endl;
                imshow("test_roi_image ", test_roi_image);
                imshow("test_image_doing", image_test_origin);

                key = waitKey(0);
                if (key == 27)  // ESC key to quit
                    break;
                temp_test++;
            }
        }
    }
    mySampleFile.close();    
    myTrainingFile.close();
    myTestFile.close();

    cout<<endl<<"Training complete."<<endl;
    waitKey();
    destroyAllWindows();
    return 0;
}