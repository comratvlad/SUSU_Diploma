#include <iostream>
#include <ctime>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <cv.h>

#include <luaT.h>
#include <TH.h>

#include "c_f_common.h"

#define PI 3.14159265

extern "C" {

using namespace std;
using namespace cv;

int n_channels = 1;
int n_points = 8;
float track_width = 260.0;
float track_height = 260.0;
float back_width = 640.0;
float back_height = 480.0;

float cur_track_width = 260.0;
float cur_track_height = 260.0;

VideoCapture cap(0);
std::vector<cv::Point2f> track_points(n_points);
Rect track_rect((back_width - track_width) / 2.0, (back_height - track_height) / 2.0, track_width, track_height);
bool not_first_frame = false;

float face_height = 100.0;
Point2f bottom_point;
Point2f left_eye;
Point2f right_eye;
Point2f top_point;

// Инициализация данных выборки 300wlp
int c_init(lua_State *L)
{
    return 0;
}

int c_get_track_rect(lua_State *L)
{
    const THFloatTensor* const track_img  = static_cast<THFloatTensor*>(luaT_toudata(L, 1, "torch.FloatTensor"));
    const THFloatTensor* const back_img  = static_cast<THFloatTensor*>(luaT_toudata(L, 2, "torch.FloatTensor"));

    // Получаем кадр с камеры
    cv::Mat src;
    cap >> src;

    cvtColor(src, src, CV_BGR2GRAY);
    src.convertTo(src, CV_32FC1);
    
    // Приводим к размеру, подаваемому в сеть
    Mat track_image;
    Size size(track_width, track_height);
    resize(src(track_rect), track_image, size, 0, 0, INTER_AREA);

    // Возвращаем область для трекинга и все изображение
    write_cv_mat2tensor(track_image, track_img);
    write_cv_mat2tensor(src, back_img);
    
    return 0;
}

int c_get_res(lua_State *L)
{
    const THFloatTensor* const input_pts  = static_cast<THFloatTensor*>(luaT_toudata(L, 1, "torch.FloatTensor"));
    const THFloatTensor* const output_pts  = static_cast<THFloatTensor*>(luaT_toudata(L, 2, "torch.FloatTensor"));
    std::vector<cv::Point2f> prev_points(n_points);
    prev_points = track_points;

    // Получаем результат работы сети
    write_tensor2vector_point(input_pts, track_points);


    // Смещаем так, чтобы координаты точек были относительно всего изображения
    for (int i = 0; i < track_points.size(); i++) {
        track_points.at(i).x *= cur_track_width;
        track_points.at(i).y *= cur_track_height;
        track_points.at(i).x += track_rect.x;
        track_points.at(i).y += track_rect.y;
    }

    // Считаем высоту лица
    bottom_point = (track_points[7] + track_points[6]) / 2.0;
    left_eye = (track_points[3] + track_points[2]) / 2.0;
    right_eye = (track_points[4] + track_points[5]) / 2.0;
    top_point = (left_eye + right_eye) / 2.0;
    float face_height = sqrt((top_point.x - bottom_point.x) * (top_point.x - bottom_point.x) +
                             (top_point.y - bottom_point.y) * (top_point.y - bottom_point.y));


    // Вырезаем новую область интересов
    if (not_first_frame) {

        // Параметры новой области значительно больше высоты лица т.к. :
        // 1 - полученные из сети точки характеризуют положение лица, а в обучающей выборке участвует голова целиком
        // 2 - в обучающей выборке изображение головы масштабировалось, поэтому отношение ее размера в кадре дожно быть 
        //     значительно меньше самого кадра
        float new_track_width = face_height * 5.5;
        float new_track_height = face_height * 5.5;

        if (new_track_width <= 480.0 && new_track_height <= 480.0) {
            cur_track_width = new_track_width;
            cur_track_height = new_track_height;
        } else {
            cur_track_width = 480.0;
            cur_track_height = 480.0;
        }

        float new_track_x = top_point.x - cur_track_width / 2.0;
        float new_track_y = top_point.y - cur_track_height / 2.0 + face_height/2.0;

        if (new_track_x >= 0 && new_track_x <= (back_width - cur_track_width) &&
            new_track_y >= 0 && new_track_y <= (back_height - cur_track_height)) {
            track_rect.x = new_track_x;
            track_rect.y = new_track_y;
            track_rect.width = cur_track_width;
            track_rect.height = cur_track_height;
        }
    }

    not_first_frame = true;

    // Возвращаем
    write_vector_point2tensor(track_points, output_pts);
    lua_pushnumber(L, bottom_point.x);
    lua_pushnumber(L, bottom_point.y);

    return 2;
}

// Регистрация функция для использования их в lua-коде
int luaopen_ctracking(lua_State *L)
{

    lua_register(
            L,
            "c_init",
            c_init
    );

    lua_register(
            L,
            "c_get_track_rect",
            c_get_track_rect
    );

    lua_register(
            L,
            "c_get_res",
            c_get_res
    );

    return 0;
}
} // extern "C"
