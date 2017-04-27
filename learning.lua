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
require 'optim'
require 'nn'

require 'cudnn'
require 'cunn'
require 'cutorch'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor');

-- Инициализация выборки
local count, test_count = c_init()
print('Данные выборки загружены...')

-- Параметры обучения
local batchsize = 16
local rows = 210
local cols = 210
local epoch = 100
local net = model.createSimple300wlpCNN()
local criterion = nn.MSECriterion()
local optimState = {
    learningRate = 0.1,
    weightDecay = 1e-6,
    momentum = 0.9
}

-- Переходим под cuda
net:cuda()
criterion:cuda()

local parameters, gradParameters = net:getParameters()

-- -- Выделение памяти
local input = torch.FloatTensor(batchsize, 1, rows, cols)
local target = torch.FloatTensor(batchsize, 2, 68)
local input_c = torch.CudaTensor(batchsize, 1, rows, cols)
local target_c = torch.CudaTensor(batchsize, 2, 68)
local result = torch.FloatTensor(batchsize, 2, 68)

print('Начинаем обучение...')

-- Обучение
for it=1,epoch do

    print('Эпоха = ' .. it)

    -- Подготовление данных выборки (в т.ч. аугментация)
    c_prepare_iteration()

    xlua.progress(0, count)

    -- Идем батчами по выборке
    for i=1,count,batchsize do

        -- Получение подготовленных данных выборки
        for j=1,batchsize do
            c_get_data(i, j, input[j], target[j])
        end

        -- Нормализация обучающих данных
        input:div(255.0)
        target:div(210.0)

        -- Копирование в cuda-буфер
        input_c:copy(input)
        target_c:copy(target)

        -- Непосредственно обучение
        local feval = function(x)

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local outputs = net:forward(input_c)

            local loss = criterion:forward(outputs, target_c)
            local dloss_doutputs = criterion:backward(outputs, target_c)
            net:backward(input_c, dloss_doutputs)

            return loss, gradParameters
        end

        optim.sgd(feval, parameters, optimState)

        xlua.progress(i + batchsize - 1, count)

    end

    xlua.progress(count, count)

    -- Тест
    xlua.progress(0, test_count)

    local test_result = 0.0

    for i=1,test_count,batchsize do

        -- Получение подготовленных данных выборки
        for j=1,batchsize do
            c_get_test_data(i, j, input[j], target[j])
        end

        input:div(255.0)
        input_c:copy(input)

        local result_c = net:forward(input_c)
        result_c:mul(210.0)
        result:copy(result_c)

        for j=1,batchsize do
            test_result = test_result + dfun.e_norm(result[j], target[j])
        end

        xlua.progress(i + batchsize - 1, test_count);

    end

    xlua.progress(test_count, test_count)

    print('Результат тестирования = ', test_result)

    collectgarbage();

    -- Основываясь на результатах теста - меняем шаг
    print('Новый шаг:')
    new_rate = io.read()
    if (new_rate == 'end') then break end
    optimState = {
        learningRate = tonumber(new_rate),
        weightDecay = 1e-6,
        momentum = 0.9
    }

end

-- Сохранение результатов
print('Обучение завершено, сохранять результат?')
local answer = io.read()
if (answer == 'yes') then
    torch.save('net/simple300wlpVer22CNN.t7', net)
end
print('Готово')
