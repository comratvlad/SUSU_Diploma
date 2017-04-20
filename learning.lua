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

require 'cunn'
require 'cutorch
require 'cudnn'

local count, test_count = c_init()
print('Данные выборки загружены...')
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
local parameters,gradParameters = net:getParameters()


print('Начинаем обучение...')

-- Обучение
for it=1,epoch do

    print('Эпоха = ' .. it)

    -- Подготовление данных выборки (в т.ч. аугментация)
    c_prepare_iteration()

    -- Идем батчами по выборке
    for i=1,count,batchsize do

        -- Выделение памяти
        local input = torch.FloatTensor(batchsize, 1, rows, cols)
        local target = torch.FloatTensor(batchsize, 2, 68)

        -- Получение подготовленных данных выборки
        for j=1,batchsize do
            c_get_data(i, j, input[j], target[j])
        end

        input =input:double()/255.0
        target = target:double()/210.0

        --for j=1,batchsize do
        --   dfun.showPointedPic(input[j], target[j]*210.0)
        --   io.read()
        --end

        -- Непосредственно обучение
        local feval = function(x)

            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local outputs = net:forward(input)

            local loss = criterion:forward(outputs, target)
            local dloss_doutputs = criterion:backward(outputs, target)
            net:backward(input, dloss_doutputs)

            return loss, gradParameters
        end

        optim.sgd(feval, parameters, optimState)

    end

    -- Тест
    local test_result = 0.0
    for i=1,test_count,batchsize do

        -- Выделение памяти
        local input = torch.FloatTensor(batchsize, 1, rows, cols)
        local target = torch.FloatTensor(batchsize, 2, 68)

        -- Получение подготовленных данных выборки
        for j=1,batchsize do
            c_get_test_data(i, j, input[j], target[j])
        end

        local result = net:forward(input:double()/255.0)*210.0

        for j=1,batchsize do
            test_result = test_result + dfun.e_norm(result[j], target:double()[j])
        end
    end
    print('Результат тестирования = ', test_result)

    -- Основываясь на результатах теста - меняем шаг
    print('Новый шаг:')
    new_rate = io.read()
    if (new_rate == 'end') then break end
    optimState = {learningRate = tonumber(new_rate),
        weightDecay = 1e-6,
        momentum = 0.9
    }

end

-- Сохранение результатов
print('Обучение завершено, сохранять результат?')
local answer = io.read()
if answer == 'yes' then torch.save('net/simple300wlpVer22CNN.t7', net) end
print('Готово')
