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
local n_points = 4
local track_width = 260
local track_height = 260
local back_width = 640
local back_height = 480

local net = torch.load('net/deepTrack300wlpVer3CNN.t7')
net:evaluate()

local track_rect = torch.FloatTensor(n_cnannels, track_width, track_height)
local c_input = torch.CudaTensor(n_cnannels, track_width, track_height)
local output = torch.FloatTensor(2, n_points)
local real_points = torch.FloatTensor(2, n_points)
local back = torch.FloatTensor(n_cnannels, back_height, back_width)

while true do

    -- Получаем область для трекинга и изображение целиком
    c_get_track_rect(track_rect, back)

    -- Получаем результат работы сети - 4 точки лица
    track_rect:div(255.0)
    c_input:copy(track_rect)
    c_output = net:forward(c_input)
    c_output:mul(track_width)
    output:copy(c_output)

    -- Отправляем результат и по нему определяем (в Си) новую область для трекинга
    -- также вычисляем real_points - положение точек лица в координатах всего изображения
    track_x, track_y = c_get_res(output, real_points)

    -- Отображаем результат
    local x1 = real_points[1][1]
    local y1 = real_points[2][3]
    local x2 = real_points[1][2]
    local y2 = real_points[2][4]
    back:div(255.0)
    result_frame = dfun.getRectangledPic(back, x1, y1, x2, y2)
    --result_frame_1 = dfun.getRectangledPic(result_frame, track_x, track_y, track_x + track_height, track_y + track_height, {lineWidth = 1, color = {255, 0, 0}})
    --result_frame = dfun.getPointedPic(result_frame, real_points, {color = {255, 0, 0},  size = 2 })
    d = image.display{image=result_frame, win=d, zoom=1 }

end
