#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace std;

vector<cv::Mat> images;
vector<vector<cv::Point2f>> points;

const string data_path = "/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/data(finished)/";

void push_data() {
    // Чтение файла с данными о путях к файлам
    ifstream config(data_path + "/config.txt");
    //ofstream out_im("/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/images.bat",ios::binary|ios::out); 
    //ofstream out_pts("/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/points.bat",ios::binary|ios::out); 

    //int s = 0;
    int n; // Кол-во оригинальных изображений
    config >> n;
    // Цикл по оригинальным изображениям
    for (int i = 0; i < 5; i++) {
        int k; // Кол-во синтезированных изображений
        config >> k;
        // Цикл по синтезированным изображениям
        for (int j = 0; j < k; j++) {
            string im_path, point_3d_path;
            config >> im_path;          // Путь до изображения
            config >> point_3d_path;    // Путь до файла с разметкой (3d)

            cv::Mat img = cv::imread(data_path + im_path);
            std::ifstream file(data_path + point_3d_path);
            vector<cv::Point2f> cur_points;
            for(int i1 = 0; i1 < 68; ++i1)
            {
                float x = 0, y = 0;
                file >> x >> y;
                //circle(img, cv::Point2f(x, y), 2, cv::Scalar(0, 0, 255), -1);
                cur_points.push_back(cv::Point2f(x, y));
            }
            file.close();

            //out_im.write((char*)&img,sizeof img);
            //out_pts.write((char*)&cur_points,sizeof cur_points);
            images.push_back(img);
            points.push_back(cur_points);
            //cv::imshow("img", img);
            //while(27 != (uchar) cv::waitKey());
        }
        //s++;
        //cout << s << endl;
    }
    //out_im.close(); //Закрыли открытый файл
    //out_pts.close();
}

int main() {
    srand(time(0));
    push_data();
    cout << "runtime = " << clock()/1000.0 << endl;
    return 0;
}
