#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main( ) {
    int x1, y1, x2, y2, w, h, temp=0, key, contour_size;
    String name;
    Mat image;          image = imread("image-data.JPG" ,1); 

    Mat blur_image;Mat thresh_image;vector<vector<Point>> contour;
    vector<Vec4i> hierarchy;Mat canny_image;
    GaussianBlur(image,blur_image,Size(3,3),11);
    threshold(blur_image, thresh_image, 150, 255, THRESH_BINARY_INV);        // imshow("thresh", thresh_image);
    Canny(thresh_image, canny_image, 20, 200,5);                             // imshow("canny", canny_image);
    findContours(canny_image, contour, hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    contour_size = contour.size();
    Mat roi_image, roismall_image, sample_image;
    for (int i=0; i< contour_size; i++){
        if (contourArea(contour[i])>50){
            vector<vector<Point> > contours_poly( contour_size );
            approxPolyDP( contour[i], contours_poly[i], 3, true );
            vector<Rect> boundRect(contour_size );                          // bound = rang buoc
            boundRect[i] = boundingRect( contours_poly[i] );
            x1 = boundRect[i].tl().x;
            y1 = boundRect[i].tl().y;
            x2 = boundRect[i].br().x;
            y2 = boundRect[i].br().y;
            w = x2 - x1;
            h = y2 - y1;
            if (h>15){
                // draw rectangle
                rectangle( image, boundRect[i].tl(), boundRect[i].br(),Scalar(0,255,0),2);
                Rect rect1(x1+1,y1+1,w-1,h-1);
                roi_image = image(rect1);                                    // crop origin image      //crop = cut
                resize(roi_image, roismall_image, Size(30,30), 0, 0);

                imshow("roismall_image ", roismall_image);
                imshow( "image-doing", image );

                key = waitKey(0);
                switch (key){
                    case 27:
                        break;
                    case 48:    case 176:   name = format("image-data/0/%d.jpg",temp);  break;
                    case 49:    case 177:   name = format("image-data/1/%d.jpg",temp);  break;
                    case 50:    case 178:   name = format("image-data/2/%d.jpg",temp);  break;
                    case 51:    case 179:   name = format("image-data/3/%d.jpg",temp);  break;
                    case 52:    case 180:   name = format("image-data/4/%d.jpg",temp);  break;
                    case 53:    case 181:   name = format("image-data/5/%d.jpg",temp);  break;
                    case 54:    case 182:   name = format("image-data/6/%d.jpg",temp);  break;
                    case 55:    case 183:   name = format("image-data/7/%d.jpg",temp);  break;
                    case 56:    case 184:   name = format("image-data/8/%d.jpg",temp);  break;
                    case 57:    case 185:   name = format("image-data/9/%d.jpg",temp);  break;
                    default:
                        break;
                }                
                imwrite(name, roi_image);
                cout<<temp<<" roi size: "<<roi_image.size()<<"  key: "<<key<<endl;
                temp++;
            }
        }
    }
    waitKey(0);
    destroyAllWindows();
    return 0;
}