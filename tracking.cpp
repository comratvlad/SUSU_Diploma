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

extern "C" {

using namespace std;
using namespace cv;

int n_channels = 1;
int n_points = 4;
float track_width = 260.0;
float track_height = 260.0;
float back_width = 640.0;
float back_height = 480.0;

VideoCapture cap(0);
std::vector<cv::Point2f> track_points(4);
Rect track_rect((back_width - track_width) / 2.0, (back_height - track_height) / 2.0, track_width, track_height);
bool not_first_frame = false;

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

    // Возвращаем область для трекинга и все изображение
    write_cv_mat2tensor(src(track_rect), track_img);
    write_cv_mat2tensor(src, back_img);

    // Вырезаем новую область для трекинга
    float cur_track_width = track_points[1].x - track_points[0].x;
    float cur_track_height = track_points[3].y - track_points[2].y;
    float new_track_x = track_points[0].x - ((track_width - cur_track_width) / 2.0);
    float new_track_y = track_points[2].y - ((track_height - cur_track_height) / 2.0);
    if (not_first_frame && new_track_x >= 0 && new_track_x <= (back_width - track_width) &&
                          new_track_y >= 0 && new_track_y <= (back_height - track_height))
    {
        track_rect.x = new_track_x;
        track_rect.y = new_track_y;
    }
    not_first_frame = true;
    
    return 0;
}

int c_get_res(lua_State *L)
{
    const THFloatTensor* const input_pts  = static_cast<THFloatTensor*>(luaT_toudata(L, 1, "torch.FloatTensor"));
    const THFloatTensor* const output_pts  = static_cast<THFloatTensor*>(luaT_toudata(L, 2, "torch.FloatTensor"));

    // Получаем результат работы сети
    write_tensor2vector_point(input_pts, track_points);

    // Смещаем так, чтобы координаты точек были относительно всего изображения
    for (int i = 0; i < track_points.size(); i++) {
        track_points.at(i).x += track_rect.x;
        track_points.at(i).y += track_rect.y;
    }

    // Возвращаем
    write_vector_point2tensor(track_points, output_pts);

    lua_pushnumber(L, track_rect.x);
    lua_pushnumber(L, track_rect.y);

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
