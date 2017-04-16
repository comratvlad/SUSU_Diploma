-- =======================================================
-- model.lua
--
-- Модуль конструирования нейронных сетей, для
-- распознания антропометрических точек лица
-- Facial Keypoints Detection и 300wlp
--
-- =======================================================

require 'torch'
require 'constructor'

model = {}

-- Первый вариант сети
function model.createSimpleCNN()
  
  net = nn.Sequential()

  net:add(nn.SpatialConvolutionMM(1, 16, 3, 3))

  net:add(nn.ReLU(false))
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  net:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
  net:add(nn.ReLU(false))
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  net:add(nn.View(32*22*22))
  net:add(nn.Linear(32*22*22, 200))
  net:add(nn.ReLU(false))
  net:add(nn.Linear(200, 30))
  net:add(nn.View(2,15))
  
  return net
  
end
---------------------------------------------------------

-- Второй вариант сети
function model.makeDeepCNN(mm)

	mm = mm or 1

	local result = {}
    local iv = inception_variants(mm);
  
	result.network = construct_structure({'seq',
      {'conv', 1, 4, 3, 3, 1, 1, 0, 0}, {'activation', 4},
      {'conv', 4, 8, 3, 3, 1, 1, 0, 0}, {'activation', 8},
      --{'maxpool', 2, 2,  2, 2,  1, 1},
      
      {'conv', 8, 16, 3, 3, 1, 1, 0, 0}, {'activation', 16},
      {'maxpool', 2, 2,  2, 2,  1, 1},
      
      {'conv', 16, 32, 3, 3, 1, 1, 0, 0}, {'activation', 32},
      {'maxpool', 2, 2,  2, 2,  1, 1},
      --32x24x24
      
      iv.inception1ab,  -- {'wv'},  -- 32x24x24
      iv.inception2,    -- {'wv'},  -- 64x12x12
      iv.inception3ab,  -- {'wv'},  -- 64x12x12
      iv.inception4,    -- {'wv'},  -- 128x6x6
      iv.inception5abc, -- {'wv'},  -- 128x6x6

        {'conv', 128*mm, 512, 6, 6, 1, 1, 0, 0}, {'activation', 512},
        -- {'wv'}, -- 128x1x1
    })

    result.network:add(nn.View(512))
    result.network:add(nn.Dropout(0.4))

    result.network:add(nn.Linear(512, 256));
    result.network:add(nn.View(16));
    result.network:add(nn.Normalize(2));
    result.network:add(nn.View(256));

    result.network:add(nn.Linear(256, 128));
    result.network:add(nn.BatchNormalization(128));
    result.network:add(nn.ReLU(false))

    result.network:add(nn.Linear(128, 30));
    result.network:add(nn.View(2, 15));
  
    local x = torch.Tensor():randn(8, (SOURCE_CHANNELS_COUNT or 1), 96, 96)

	result.input_size = drop_minibatch_size( x:size() )
	result.output_size = drop_minibatch_size( result.network:forward(x):size() )

	return result;
end
---------------------------------------------------------

-- Первый вариант сети для выборки 300wlp
function model.createSimple300wlpCNN()

    net = nn.Sequential()

    net:add(nn.SpatialConvolutionMM(1, 4, 3, 3))
    net:add(nn.SpatialBatchNormalization(4))
    net:add(nn.ReLU(false))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    net:add(nn.SpatialConvolutionMM(4, 8, 3, 3))
    net:add(nn.SpatialBatchNormalization(8))
    net:add(nn.ReLU(false))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    net:add(nn.SpatialConvolutionMM(8, 16, 3, 3))
    net:add(nn.SpatialBatchNormalization(16))
    net:add(nn.ReLU(false))
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    net:add(nn.View(16*24*24))
    net:add(nn.Dropout(0.4))

    net:add(nn.Linear(16*24*24, 4096))
    net:add(nn.BatchNormalization(4096))
    net:add(nn.ReLU(false))

    net:add(nn.Linear(4096, 512))
    net:add(nn.BatchNormalization(512))
    net:add(nn.ReLU(false))

    net:add(nn.Linear(512, 2*68))
    net:add(nn.View(2,68))

    return net

end
---------------------------------------------------------

-- Второй вариант сети для выборки 300wlp
function model.makeDeep300wlpCNN(mm)

    mm = mm or 1

    local result = {}
    local iv = inception_variants(mm);

    result.network = construct_structure({'seq',
        {'conv', 1, 4, 3, 3, 1, 1, 0, 0}, {'activation', 4},
        {'conv', 4, 8, 3, 3, 1, 1, 0, 0}, {'activation', 8},
        --{'maxpool', 2, 2,  2, 2,  1, 1},

        {'conv', 8, 16, 3, 3, 1, 1, 0, 0}, {'activation', 16},
        {'maxpool', 2, 2,  2, 2,  1, 1},

        {'conv', 16, 32, 3, 3, 1, 1, 0, 0}, {'activation', 32},
        {'maxpool', 2, 2,  2, 2,  1, 1},
        --32x24x24

        iv.inception1ab,  -- {'wv'},  -- 32x24x24
        iv.inception2,    -- {'wv'},  -- 64x12x12
        iv.inception3ab,  -- {'wv'},  -- 64x12x12
        iv.inception4,    -- {'wv'},  -- 128x6x6
        iv.inception5abc, -- {'wv'},  -- 128x6x6

        {'conv', 128*13*13*mm, 512, 6, 6, 1, 1, 0, 0}, {'activation', 512},
        -- {'wv'}, -- 128x1x1
    })

    result.network:add(nn.View(512))
    result.network:add(nn.Dropout(0.4))

    result.network:add(nn.Linear(512, 256));
    result.network:add(nn.View(16));
    result.network:add(nn.Normalize(2));
    result.network:add(nn.View(256));

    result.network:add(nn.Linear(256, 128));
    result.network:add(nn.BatchNormalization(128));
    result.network:add(nn.ReLU(false))

    result.network:add(nn.Linear(128, 2*68));
    result.network:add(nn.View(2, 68));

    local x = torch.Tensor():randn(16, (SOURCE_CHANNELS_COUNT or 1), 210, 210)

    result.input_size = drop_minibatch_size( x:size() )
    result.output_size = drop_minibatch_size( result.network:forward(x):size() )

    return result;
end
---------------------------------------------------------

return model
