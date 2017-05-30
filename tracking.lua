-- =========================================================
-- tracking.lua
--
--
-- =========================================================

require 'ctracking'
require 'constructor'
require 'model'
require 'dfun'
require 'torch'
require 'optim'
require 'nn'

require 'cunn'
require 'cutorch'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

-- =========================================================

c_init()

local n_cnannels = 1
local n_points = 8
local track_width = 260
local track_height = 260
local back_width = 640
local back_height = 480

local net = torch.load('net/deepTrack300wlpVer5(01)CNN.t7')
net:evaluate()

local track_rect = torch.FloatTensor(n_cnannels, track_width, track_height)
local c_input = torch.CudaTensor(n_cnannels, track_width, track_height)
local output = torch.FloatTensor(2, n_points)
local real_points = torch.FloatTensor(2, n_points)
local back = torch.FloatTensor(n_cnannels, back_height, back_width)
local face_point = torch.FloatTensor(2, 1)

while true do

    -- Получаем область для трекинга и изображение целиком
    c_get_track_rect(track_rect, back)

    -- Получаем результат работы сети - 4 точки лица
    track_rect:div(255.0)
    c_input:copy(track_rect)
    c_output = net:forward(c_input)
    output:copy(c_output)

    -- Отправляем результат и по нему определяем (в Си) новую область для трекинга
    -- также вычисляем real_points - положение точек лица в координатах всего изображения
    track_x, track_y = c_get_res(output, real_points)

    d1 = image.display{image=track_rect, win=d1, zoom=1 }
    
    -- Отображаем результат
    local face_width = real_points[1][8] - real_points[1][3]
    local face_height = real_points[2][8] - real_points[2][3]
    local x1 = real_points[1][3] - face_height / 2.0
    local y1 = real_points[2][3] - face_height / 2.0
    local x2 = real_points[1][8] + face_height / 2.0
    local y2 = real_points[2][8] + face_height / 2.0
    back:div(255.0)

    result_frame = dfun.getRectangledPic(back, x1, y1, x2, y2)
    --result_frame = dfun.getPointedPic(back, real_points, {color = {255, 0, 0},  size = 2 })
    
    d = image.display{image=result_frame, win=d, zoom=1 }

end
