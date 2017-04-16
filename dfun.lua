-- =======================================================
-- dfun.lua
--
-- Набор вспомогательных lua функций для работы
--
-- =========================================================

local image = require 'image'
local torch = require 'torch'

dfun = {}

-- Создание изображения pic (3xWxH) с нанесенными точками
-- из таблицы или Tensor-a points (2xNUM_POINTS)
-- возвращает pic
-- opt - таблица из color и size точек, например
-- opt = {color = {0, 255, 0},  size = 1}
function dfun.getPointedPic(pic, points, opt)

    -- По умолчанию - зеленые точки
    opt = opt or {color = {0, 255, 0},  size = 1 }

    -- Получение параметров изображения и валидация
    assert(pic:nDimension() == 3, 'Wrong format of picture')
    local nChannels = pic:size()[1]
    local rows = pic:size()[2]
    local cols = pic:size()[3]
    assert(nChannels == 1 or nChannels == 3, 'Wrong format of picture')

    -- Для функций пакета image необходимо трехканальное изображение
    c_pic = torch.Tensor(3, rows, cols)

    -- Преобразуем ч-б в трехканальное, если необходимо
    if nChannels == 1 then
        c_pic[1] = pic
        c_pic[2] = pic
        c_pic[3] = pic
    end

    -- Если points - Tensor преобразуем в таблицу
    if (type(points) ~= 'table') then
        points = torch.totable(points)
    end

    -- Валидация poinst
    assert(#points == 2 , 'Wrong format of points')
    assert(#points[1] == #points[2], 'Wrong format of points')

    -- Кол-во точек
    num_points = #points[1]

    -- Нанесение точек
    for i=1,num_points do
        local x = points[1][i] - 5
        local y = points[2][i] - 5
        if (x < rows and x > 0 and y < cols and y > 0) then
            c_pic = image.drawText(c_pic, '.', x, y, opt)
        end
    end

    -- Возвращение изображения
    return c_pic

end

-- Вывод на экран изображения pic (3xWxH) с нанесенными точками
-- из таблицы или Tensor-a points (2xNUM_POINTS)
function dfun.showPointedPic(pic, points, opt)

    pic = dfun.getPointedPic(pic, points, opt)
    image.display(pic)

end

-- Сохранение в filepath изображения pic (3xWxH) с нанесенными точками
-- из таблицы или Tensor-a points (2xNUM_POINTS)
function dfun.savePointedPic(filepath, pic, points, opt)

    pic = dfun.getPointedPic(pic, points, opt)
    image.save(filepath, pic)

end

-- Уменьшение изображения pic (3xWxH) с нанесенными точками
-- из таблицы или Tensor-a points (2xNUM_POINTS) со сдвигом точек
-- до размеров width x height
-- возвращает пару pic, points (новые)
function dfun.cutPic(width, height, pic, points)

    -- TODO: исправить
    -- Не баг, а фича
    width = width - 1
    height = height - 1

    -- Нахождение угловых точек
    local xmin = torch.min(points[1])
    local xmax = torch.max(points[1])
    local ymin = torch.min(points[2])
    local ymax = torch.max(points[2])

    -- Ширина и высота лица (минимально допустимые)
    local face_width = xmax - xmin
    local face_height = ymax - ymin

    -- Провека - вдруг они больше требуемых
    assert(face_width <= width and face_height <= height, "Wrong picture")

    -- Вычисление угловых точек, по которым будем вырезать изображение
    local d_width = width - face_width
    local d_height = height - face_height
    local x1 = xmin - d_width/2.0
    local y1 = ymin - d_height/2.0
    local x2 = xmax + d_width/2.0
    local y2 = ymax + d_height/2.0

    -- Вырезаем изображение
    local respic = pic[{{},{y1, y2},{x1, x2}}]

    -- Сдвигаем точки
    local respoints = points
    respoints[1] = respoints[1] - x1
    respoints[2] = respoints[2] - y1

    return respic, respoints

end

return dfun
