-- =========================================================
-- learning.lua
--
-- Основной модуль обучения cnn для распознавания
-- антропометрических точек лица.
-- Для обучения используется выборка 300wlp
--
-- =========================================================

require 'data300wlp'
require 'constructor'
require 'model'
require 'dfun'
require 'torch'

local count = c_init()
local batchsize = 16
local rows = 210
local cols = 210
local epoch = 1

-- Checking
local input11 = torch.FloatTensor(1, rows, cols)
local target11 = torch.FloatTensor(2, 68):fill(0)
hello(input11, target11)

local input1 = torch.FloatTensor(3, rows, cols)
local target1 = target11
input1[1] = input11/255.0
input1[2] = input11/255.0
input1[3] = input11/255.0

dfun.showPointedPic(input1, target1)
-- =========================================================

-- Обучение
for it=1,epoch do

    -- Подготовление данных выборки (в т.ч. аугментация)
    c_prepare_iteration()

    -- Идем батчами по выборке
    for i=1,count,batchsize do

        -- Выделение памяти
        local input = torch.FloatTensor(batchsize, 1, rows, cols)
        local target = torch.FloatTensor(batchsize, 2, 68)

        -- Получение подготовленных данных выборки
        c_get_data(i, batchsize, input, target)

        -- Непосредственно обучение

    end
end

