# Дипломный проект
Файлы:
1) learning.lua — основной модуль обучения cnn;
2) data300wlp.cpp — модуль аугментации и работы с данными при помощи OpenCV;
3) load_data.cpp — создание бинарного файла, удобного для хранения и чтения выборки;
4) constructor.lua — модуль для конструирования cnn в более удобном и читаемом виде;
5) model.lua — создание структур тех cnn, что используются в работе на данный момент;
6) imfun.h/imfun.cpp — набор вспомогательных cpp функций для работы;
7) 300WLP_cut.lua — lua скрипт по созданию облегченной версии выборки 300WLP;
8) dfun.lua — набор вспомогательных lua функций для работы;
9) tracking.lua и tracking.cpp- работа трекинга лиц;
10) c_f_common.h - конвертация объектов OpenCV/Torch.

Небольшой комментарий: суть проекта - трекинг объектов (лиц), основанный на работе CNN,способной восстанавливать специальные точки объекта на сцене (антропометрические точки лица). Сначала работа велась по выборке: https://www.kaggle.com/c/facial-keypoints-detection. Потом взял базу пошире (раз в 10): http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.
