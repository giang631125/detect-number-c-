#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv::ml;
using namespace cv;

#define size_image 5
#define num_image 2
void convert_to_1d(Mat_<float> Mat_in, Mat_<float> Mat_out)
{
    for(int i=0; i<size_image; i++)
    {
        Mat_in.row(i).copyTo(Mat_out(Rect(size_image*i, 0, size_image, 1)));
    }
}
int main()
{
    Ptr<ml::KNearest>  knn(ml::KNearest::create());
    Mat1f trainFeatures(6,4);
    trainFeatures<< 2,2,2,2,
                    3,3,3,3,
                    4,4,4,4,
                    5,5,5,5,
                    6,6,6,6,
                    7,7,7,7;

    Mat_<int> trainLabels(1,6);
    trainLabels << 2,3,4,5,6,7;

    knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
    Mat_<float> testFeature(1,4);
    testFeature<< 3,3,3,2;
    // cout<<trainFeatures<<endl;
    // cout<<trainLabels<<endl;
    // cout<<testFeature<<endl;

    int K=1;
    Mat response,dist;
    knn->findNearest(testFeature, K, noArray(), response, dist);
    cerr <<"kq1"<< response << endl;
    cerr <<"dis1"<< dist<< endl;
     

    Mat_<float> M1; M1= imread("66.jpg",0);
    resize(M1, M1, Size(size_image,size_image),0,0);
    Mat_<float> M2; M2= imread("59.jpg",0);
    resize(M2, M2, Size(size_image,size_image),0,0);
    Mat_<float> M3; M3= imread("0.jpg",0);
    resize(M3, M3, Size(size_image,size_image),0,0);
    // cout << "M1 = " << endl << M1 << endl << endl;
    // cout << "M2 = " << endl << M2 << endl << endl;
    cout<<"type: "<<M1.type()<<endl;

    Mat_<float> sample2=Mat(num_image, size_image*size_image, CV_8UC1);
    // sample2.push_back(M1);
    // sample2.push_back(M2);
    Mat_<float> sample3=Mat(0,0,CV_8UC1);    //
    // memcpy( sample3.data, sample2.data(), sample2.size()*sizeof(float) );
    sample3.push_back(M1);
    sample3.push_back(M2);
    // sample3.push_back(M2);
    int x,y;
    
    for(int i=0; i< size_image*num_image; i++)
    {
        y=i/size_image;
        x=size_image*(i%size_image);
        // Rect a=;
        sample3.row(i).copyTo(sample2(cvRect(x,y,size_image,1)));
    }
    // sample3.row(0).copyTo( sample2(Rect(0,0,5,1)) );
    // sample3.row(1).copyTo( sample2(Rect(5,0,5,1)) );
    // sample3.row(2).copyTo( sample2(Rect(10,0,5,1)) );
    // sample3.row(3).copyTo( sample2(Rect(15,0,5,1)) );
    // sample3.row(4).copyTo( sample2(Rect(20,0,5,1)) );
    // sample3.row(5).copyTo( sample2(Rect(0,1,5,1)) );
    // sample3.row(6).copyTo( sample2(Rect(5,1,5,1)) );
    // sample3.row(7).copyTo( sample2(Rect(10,1,5,1)) );
    // sample3.row(8).copyTo( sample2(Rect(15,1,5,1)) );
    // sample3.row(9).copyTo( sample2(Rect(20,1,5,1)) );

    // cout<<"sample2 "<<sample2<<endl;

    // Mat_<float> M1_1d=Mat::zeros(1,25,CV_8UC1);
    // M1.row(0).copyTo( M1_1d( Rect(0,0,5,1) ) );
    // M1.row(1).copyTo( M1_1d( Rect(5,0,5,1) ) );
    // M1.row(2).copyTo( M1_1d( Rect(10,0,5,1) ) );
    // M1.row(3).copyTo( M1_1d( Rect(15,0,5,1) ) );
    // M1.row(4).copyTo( M1_1d( Rect(20,0,5,1) ) );
    // Mat_<float> M2_1d=Mat::zeros(1,25,CV_8UC1);
    // M2.row(0).copyTo( M2_1d( Rect(0,0,5,1) ) );
    // M2.row(1).copyTo( M2_1d( Rect(5,0,5,1) ) );
    // M2.row(2).copyTo( M2_1d( Rect(10,0,5,1) ) );
    // M2.row(3).copyTo( M2_1d( Rect(15,0,5,1) ) );
    // M2.row(4).copyTo( M2_1d( Rect(20,0,5,1) ) );
    Mat_<float> M3_1d=Mat::zeros(1,25,CV_8UC1);
    // M3.row(0).copyTo( M3_1d( Rect(0,0,5,1) ) );
    // M3.row(1).copyTo( M3_1d( Rect(5,0,5,1) ) );
    // M3.row(2).copyTo( M3_1d( Rect(10,0,5,1) ) );
    // M3.row(3).copyTo( M3_1d( Rect(15,0,5,1) ) );
    // M3.row(4).copyTo( M3_1d( Rect(20,0,5,1) ) );
    convert_to_1d(M3, M3_1d);
    // cout << "M1_1d = " << endl << M1_1d << endl << endl;
    // cout << "M2_1d = " << endl << M2_1d << endl << endl;
    // cout << "M3_1d = " << endl << M3 << endl << endl;

    // Mat_<float> sample=Mat::zeros(5,55,CV_8UC1);
    vector<int> input;
    input.push_back(1);
    input.push_back(2);
    Mat_<int> respon=Mat::zeros(1,2,CV_8UC1);
    memcpy(respon.data, input.data(), input.size()*sizeof(int) );

    // M1_1d.row(0).copyTo( sample(Rect(0,0,25,1)) );
    // M2_1d.row(0).copyTo( sample(Rect(0,1,25,1)) );
    // cout << "sample = " << endl << sample << endl << endl;
    // cout << "respon = " << endl << respon << endl << endl;

    Ptr<ml::KNearest> knear(ml::KNearest::create());
    ofstream myFile;
    myFile.open("test.txt", std::ofstream::out | std::ofstream::trunc);
    myFile<<sample3<<endl;
    myFile<<sample2<<endl;
    myFile.close();
    knear->train(sample2, ml::ROW_SAMPLE, respon);
    
    Mat resu, dist2;
    knear->findNearest(M3_1d, 1, noArray(), resu, dist2);
    cout<<" kw2 "<<resu<<endl;
    cout<<" dis2 "<<dist2<<endl;
}
