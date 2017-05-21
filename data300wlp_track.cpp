// Создание обучающей выборки для трекинга изображений
// Исходные данные: выборка 300wlp (подготовленная, серая, 210x210)
//                  92 фоновых изображений (больших)
// Результат:
void create_track_data() {

    srand(time(0));

    // Чтение файла с данными о путях к файлам
    ifstream config_background(background_data_path + "/config.txt");
    ifstream config_data(data_path + "/config.txt");
    ofstream output(finished_data_path, ios::binary|ios::out);

    // Создаем вектор с фоновыми изображениями
    int background_count = 0;
    config_background >> background_count;
    vector<cv::Mat> backgrounds;
    for (int i = 0; i < background_count; i++) {
        string name;
        config_background >> name;
        cv::Mat back = cv::imread(background_data_path + name);
        cvtColor(back, back, CV_BGR2GRAY);
        backgrounds.push_back(back);
    }

    // Параметры базы
    int n;                  // Кол-во оригинальных изображений
    int back_cols = 220;
    int back_rows = 220;
    config_data >> n;

    // Цикл по оригинальным изображениям
    for (int i = 0; i < n; i++) {

        int k; // Кол-во синтезированных изображений
        config_data >> k;

        // Цикл по синтезированным изображениям
        for (int j = 0; j < k; j++) {

            string im_path, point_3d_path;
            config_data >> im_path;          // Путь до изображения
            config_data >> point_3d_path;    // Путь до файла с разметкой (3d)

            // Загружаем изображение
            cv::Mat img_i = cv::imread(data_path + im_path);

            // Преобразование к серому (если нужно)
            cv::Mat img;
            cvtColor(img_i, img, CV_BGR2GRAY);

            // Загружаем точки
            std::ifstream file(data_path + point_3d_path);
            vector<cv::Point2f> full_points;
            for(int it = 0; it < 68; ++it)
            {
                float x = 0, y = 0;
                file >> x >> y;
                full_points.push_back(cv::Point2f(x, y));
            }
            file.close();

            // Выбираем фон
            int num = rand() % background_count;
            cv::Mat back = backgrounds.at(num).clone();
            int rand_param = rand() % 5;
            if (rand_param == 1) {
                cv::resize(back, back, cv::Size(back_cols, back_rows), 0, 0);
            } else {
                int dx = rand() % (back.cols - back_cols);
                int dy = rand() % (back.rows - back_rows);
                back = back(cv::Rect(dx, dy, back_cols, back_rows));
            }

            // Случайная аугментация - поворачиваем и масштабируем элемент выборки
            float rand_rotation = rand() % 45 * pow(-1.0, rand() % 2 * 1.0);
            float rand_size = 0.4 + rand() % 6 * 0.1;
            float rand_light = 1.0 + rand() % 3 * 0.1;
            img *= rand_light;
            cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(210/2, 210/2), rand_rotation, rand_size);
            cv::warpAffine(img, img, rot, cv::Size(210, 210));
            cv::transform(full_points, full_points, rot);

            // Определяем специальные точки лица
            cv::Point2f left_point(210, 0), right_point(0, 0), top_point(0, 210), bottom_point(0, 0);
            for (int it = 0; it < full_points.size(); it++) {
                float cur_x = full_points.at(it).x;
                float cur_y = full_points.at(it).y;
                if (cur_x < left_point.x) {
                    left_point = full_points.at(it);
                }
                if (cur_y > bottom_point.y) {
                    bottom_point = full_points.at(it);
                }
                if (cur_x > right_point.x) {
                    right_point = full_points.at(it);
                }
                if (cur_y < top_point.y) {
                    top_point = full_points.at(it);
                }
            }
            cv::Point2f center_point;
            center_point.x = (left_point.x + right_point.x) / 2.0;
            center_point.y = (top_point.y + bottom_point.y) / 2.0;
            float R = (e_r(center_point, left_point) + e_r(center_point, right_point) +
                      e_r(center_point, top_point) + e_r(center_point, bottom_point)) / 4.0;

            // Выбираем случайную область фона и вставляем туда лицо
            int dx = rand() % (back.cols - img.cols);
            int dy = rand() % (back.rows - img.rows);
            cv::Rect rect(dx, dy, img.rows, img.cols);

            float norm = 0.0;
            for(int y = 0; y < img.rows; y++) {
                uchar* ptr_back = (uchar*) (back(rect).data + y * back(rect).step);
                uchar* ptr_img = (uchar*) (img.data + y * img.step);
                for (int x = 0; x < img.cols; x++) {
                    if (ptr_img[x] != 0) {
                        cv::Point2f tek(x, y);
                        float rr = e_r(center_point, tek);
                        if (rr>norm) norm = rr;
                    }

                }
            }

            for(int y = 0; y < img.rows; y++) {
                uchar* ptr_back = (uchar*) (back(rect).data + y * back(rect).step);
                uchar* ptr_img = (uchar*) (img.data + y * img.step);
                for (int x = 0; x < img.cols; x++) {
                    if (ptr_img[x] != 0) {
                        cv::Point2f tek(x, y);
                        float rr = e_r(center_point, tek);
                        //float alpha = (1.0 - 1.0 / ((rr - R)*(rr - R)))/2.0;
                        float alpha = (1.0 - (rr-R)/(norm - R))/(1.0*(1.0+1.5*(rr-R)/(norm - R)));
                        if ((int)rr <= (int)(R + 1.0)) alpha = 1.0;
                        ptr_back[x] = alpha*ptr_img[x] + (1.0-alpha)*ptr_back[x];
                    }
                }
            }

            for (int it = 0; it < full_points.size(); it++) {
                full_points.at(it) += cv::Point2f(dx,dy);
            }
            
            left_point += cv::Point2f(dx,dy);
            bottom_point += cv::Point2f(dx,dy);
            right_point += cv::Point2f(dx,dy);
            top_point += cv::Point2f(dx,dy);
            center_point += cv::Point2f(dx,dy);

            //------------------------------------------------------

            // Выбираем нужные точки
            vector<cv::Point2f> cur_points;
            cv::Point2f first_point(left_point.x, center_point.y);
            cv::Point2f second_point(right_point.x, center_point.y);
            cv::Point2f third_point(center_point.x, top_point.y);
            cv::Point2f fourth_point(center_point.x, bottom_point.y);
            cur_points.push_back(first_point);
            cur_points.push_back(second_point);
            cur_points.push_back(third_point);
            cur_points.push_back(fourth_point);

            // Записываем изображение
            const int type = back.type();
            const int rows = back.rows;
            const int cols = back.cols;
            output.write((char const*) &type, sizeof(type));
            output.write((char const*) &rows, sizeof(rows));
            output.write((char const*) &cols, sizeof(cols));
            for(int i2 = 0; i2 < rows; ++i2) {
                output.write((char const*) back.ptr(i2), back.cols * back.elemSize());
            }

            // Записываем точки
            const size_t count = cur_points.size();
            output.write((char const*) &count, sizeof(count));
            output.write((char const*) cur_points.data(), sizeof(cur_points[0]) * cur_points.size());

            back.release();
            img.release();
        }

    }

    config_background.close();
    config_data.close();
    output.close();
}
