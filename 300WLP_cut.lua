-- =======================================================
-- 300WLP_cut.lua
--
-- Обработка "сырых" данных выборки 300WLP, в том числе:
--  1) обрезка изображений до размеров 210х210;
--  2) сохранение всех данных в более компактном виде;
--
--  В данном случае взяты точки из txt3d.
--
-- =======================================================

local image = require 'image'
local torch = require 'torch'
local dfun = require 'dfun'

function string:split(sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) fields[#fields+1] = c end)
    return fields
end

-- Корневые папки - "сырая" и "готовая"
local damp_root = "/home/vladislav/300WLP/data(damp)/"
local finished_root = "/home/vladislav/300WLP/data(finished)/"

-- Соответствующие дескрипторы
local damp_f = io.open(damp_root .. "config.txt", "r")
local finished_f = io.open(finished_root .. "config.txt", "w")

-- Читаем первую строку - кол-во оригинальных изображений
-- тут же вносим её в новый конфигурационный файл
local n = damp_f:read()
finished_f:write(n)
finished_f:write('\n')

-- Читаем пустую строку "в холостую"
damp_f:read()

-- Идем по оригинальным изображениям
for i=1,n do

    -- Читаем кол-во синтезированных изображений
    local k = damp_f:read()
    local gender = damp_f:read()
    finished_f:write(k)
    finished_f:write('\n')

    -- Идем по ним
    for j=1,k do

        -- Читаем важные данные
        local im_path = damp_f:read()
        local im_flip_path = damp_f:read()
        local point_2d_path = damp_f:read()
        local point_3d_path = damp_f:read()

        -- Удаляем ведущие пробелы
        im_path = string.sub(im_path, 3, #im_path)
        --im_flip_path = string.sub(im_flip_path, 3, #im_flip_path)
        --point_2d_path = string.sub(point_2d_path, 3, #point_2d_path)
        point_3d_path = string.sub(point_3d_path, 3, #point_3d_path)

        -- Вносим данные в новый конф. файл
        finished_f:write(im_path)
        finished_f:write('\n')
        finished_f:write(point_3d_path)
        finished_f:write('\n')

        -- Читаем изображение
        local pic = image.load(damp_root .. im_path)

        -- Читаем точки
        local points = torch.FloatTensor(2, 68)
        local f_pts = io.open(damp_root .. point_3d_path, "r")
        for it=1,68 do
            local pts = f_pts:read():split(' ')
            local x_s = string.gsub(pts[1], '%.', ',')
            local y_s = string.gsub(pts[2], '%.', ',')
            local x = tonumber(x_s)
            local y = tonumber(y_s)
            points[1][it] = x
            points[2][it] = y
        end
        f_pts:close()

        -- Уменьшение изображения до 210х210 со сдвигом точек
        local new_pic, new_points = dfun.cutPic(210, 210, pic, points)

        -- Заполняем новую выборку изображением и файлом с точками
        image.save(finished_root .. im_path, new_pic)
        local finished_f_pts = io.open(finished_root .. point_3d_path, "w")
        for i1=1,68 do
            local x_n = string.gsub(tostring(new_points[1][i1]), ',', '.')
            local y_n = string.gsub(tostring(new_points[2][i1]), ',', '.')
            finished_f_pts:write(x_n .. ' ' .. y_n)
            finished_f_pts:write('\n')
        end
        finished_f_pts:close()

        -- Читаем пустую строку "в холостую"
        damp_f:read()
    end
    print(i)
end

-- Завершение работы
damp_f:close()
finished_f:close()

-- =================================
-- Итог: исходная выборка ~1.7 Гб
--       теперь -> 553.9 Мб
-- =================================
-- Warning!
-- BUG in IBUG (now fixed)
--
-- IBUG_image_092 _01_0.jpg
-- ...
-- IBUG_image_092 _01_10.jpg
--
-- instead
--
-- IBUG_image_092_01_0.jpg
-- ...
-- IBUG_image_092_01_10.jpg
-- =================================
