#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "imfun.h"

using namespace std;

// Внутреннее представление выборки 300wlp
vector<cv::Mat> images;
vector<vector<cv::Point2f>> points;

// ...
const string data_path = "/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/data(finished)/";
const string cpp_data_path = "/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/300wlp(gray).dat";
const int wlp_size = 61225;


// Запись данных в бинарный файл
// Результат: работает весьма быстро, но получившийся размер файла (в случае RGB изображений 210х210) ~8Гб
//            таких изображений вместе с точками можно загрузить лишь около 20000 (тоже относительно быстро)
//            в YUV формате ситуация не намного лучше
//            в оттенках серого файл ~2.7Гб
void write_data() {

    // Чтение файла с данными о путях к файлам
    ifstream config(data_path + "/config.txt");
    ofstream output(cpp_data_path, ios::binary|ios::out);

    int n; // Кол-во оригинальных изображений
    config >> n;
    // Цикл по оригинальным изображениям
    for (int i = 0; i < n; i++) {

        int k; // Кол-во синтезированных изображений
        config >> k;

        // Цикл по синтезированным изображениям
        for (int j = 0; j < k; j++) {

            string im_path, point_3d_path;
            config >> im_path;          // Путь до изображения
            config >> point_3d_path;    // Путь до файла с разметкой (3d)

            // Загружаем изображение
            cv::Mat img_i = cv::imread(data_path + im_path);
            // Преобразование к серому (если нужно)
            cv::Mat img;
            cvtColor(img_i, img, CV_BGR2GRAY);

            // Загружаем точки
            std::ifstream file(data_path + point_3d_path);
            vector<cv::Point2f> cur_points;
            for(int i1 = 0; i1 < 68; ++i1)
            {
                float x = 0, y = 0;
                file >> x >> y;
                cur_points.push_back(cv::Point2f(x, y));
            }
            file.close();

            // Записываем изображение
            const int type = img.type();
            const int rows = img.rows;
            const int cols = img.cols;
            output.write((char const*) &type, sizeof(type));
            output.write((char const*) &rows, sizeof(rows));
            output.write((char const*) &cols, sizeof(cols));
            for(int i2 = 0; i2 < rows; ++i2) {
                output.write((char const*) img.ptr(i2), img.cols * img.elemSize());
            }

            // Записываем точки
            const size_t count = cur_points.size();
            output.write((char const*) &count, sizeof(count));
            output.write((char const*) cur_points.data(), sizeof(cur_points[0]) * cur_points.size());

        }
        //cout << i << endl;
    }
    config.close();
    output.close();
}


// Чтение всей выборки 300wlp во внутреннее представление
void read_data() {

    // Чтение файла с данными
    ifstream input(cpp_data_path, ios::binary|ios::in);

    for (int i = 0; i < wlp_size; i++) {

        // Чтение изображения
        int type, rows, cols;
        input.read((char*) &type, sizeof(type));
        input.read((char*) &rows, sizeof(rows));
        input.read((char*) &cols, sizeof(cols));
        cv::Mat img(rows, cols, type);
        for(int j = 0; j < rows; ++j) {
            input.read((char*) img.ptr(j), cols * img.elemSize());
        }

        // Чтение точек
        size_t count;
        input.read((char*)&count, sizeof(count));
        std::vector<cv::Point2f> pts(count);
        input.read((char*)pts.data(), sizeof(pts[0]) * pts.size());

        // Все - в вектора
        images.push_back(img);
        points.push_back(pts);

        // TODO: проверка на окончание потока

        //cout << i << endl;
    }

    input.close();
}


int main() {

    //write_data();
    read_data();

    // Проверка
    for (int i = 0; i < images.size(); i ++) {
        showPointedImage(images.at(i), points.at(i));
    }
    
    return 0;
}


