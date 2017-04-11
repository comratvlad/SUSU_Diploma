#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <luaT.h>
#include <TH.h>

#include "c_f_common.h"

extern "C" {

using namespace std;

// Внутреннее представление выборки 300wlp
vector<cv::Mat> images;
vector< vector<cv::Point2f> > points;

// ...
const char cpp_data_path[] = "/home/vladislav/CLionProjects/LuaTorchProjects/DataLoadProject/300wlp(gray).dat";
const int wlp_size = 61225;

// Инициализация данных выборки 300wlp
int c_init(lua_State *L)
{

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
    }

    input.close();

    lua_pushnumber(L, wlp_size);

    return 1;
}

// Инициализация параметров аугментации
int c_prepare_iteration(lua_State *L)
{
    return 0;
}

// Передача обработанных данных выборки для обучения
int c_get_data(lua_State *L)
{
    return 0;
}

// Тест - проверим, загрузится ли какое-либо изображение с точками правильно
// передадим его в lua-код и проверим
int hello(lua_State *L)
{
    // Номер изображения
    int n = 77;

    // Определяем область памяти, куда нужно вернуть картинку
    const THFloatTensor* const output_img  = static_cast<THFloatTensor*>(luaT_toudata(L, 1, "torch.FloatTensor"));
    // Загружаем саму картинку, преобразуем к нужному типу
    cv::Mat src = images.at(n);
    src.convertTo(src, CV_32FC1);
    // Загружаем картинку в нужную область памяти
    write_cv_mat2tensor(src, output_img);

    // Определяем область памяти, куда нужно вернуть точки
    THFloatTensor* output_pts  = static_cast<THFloatTensor*>(luaT_toudata(L, 2, "torch.FloatTensor"));
    // Загружаем сами точки
    vector<cv::Point2f> pts = points.at(n);
    // Загружаем вектор точек в нужную область памяти
    // write_vector_point2tensor - функция, добавленная в c_f_common.h
    // ее код закомментирован в конце файла
    write_vector_point2tensor(pts, output_pts);

    return 0;
}

// Регистрация функция для использования их в lua-коде
int luaopen_data300wlp(lua_State *L)
{

    lua_register(
            L,
            "c_init",
            c_init
    );

    lua_register(
            L,
            "c_prepare_iteration",
            c_prepare_iteration
    );

    lua_register(
            L,
            "c_get_data",
            c_get_data
    );

    lua_register(
	     L,
	     "hello",
             hello
    );

    return 0;
}

} // extern "C"

/*
inline
void write_vector_point2tensor(
        const std::vector<cv::Point2f> &src,
        THFloatTensor const* const dst)
{
    RAssert(!src.empty());

    RAssert(dst);
    assertContinuous(dst);
    RAssert(dst->nDimension == 2);

    const int num_pts = src.size();

    RAssert(dst->size[0] == 2);
    RAssert(dst->size[1] == num_pts);

    float* const dst_ptr = dst->storage->data + dst->storageOffset;

    for(int i = 0; i < num_pts; ++i)
    {
        dst_ptr[i] = src.at(i).x;
        dst_ptr[i + num_pts] = src.at(i).y;
    }
}
 */
