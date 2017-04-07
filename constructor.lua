-- ==================================
-- constructor.lua
--
-- Модуль для более удобного и читаемого создания CNN
--
-- ==================================

require 'torch' -- torch
require 'nn'
require 'cunn'
require 'cutorch'

require 'cudnn';
do
	CUDNN_AVAILABLE = false;
	local CUDNN_ERROR;
	CUDNN_AVAILABLE, CUDNN_ERROR = pcall(require, 'cudnn');

	if not CUDNN_AVAILABLE then
		print(
			'\n\nWARNINGN!\n',
			'CUDNN not available!\n',
			'error: \'', CUDNN_ERROR , '\'',
			'\nWARNINGN!\n\n');
	else
		assert(cudnn);

		cudnn.fastest = true;
		cudnn.benchmark = true;
	end
end





function drop_minibatch_size(sizes)
	assert(sizes:size() >= 2);
	local res = torch.LongStorage(sizes:size() - 1);
	for i = 2, sizes:size() do
		res[i - 1] = sizes[i];
	end
	return res;
end



function makeActivationUnit(input_channels)
	local res = nn.Sequential();

	res:add(nn.SpatialBatchNormalization(
		input_channels,  -- nFeature
		1e-5,            -- eps (default - 1e-5)
		0.1,             -- momentum (default - 0.1)
		true             -- affine (default - true)
		));

	res:add(nn.ReLU(false));

	return res;
end


function makeSpatialConvolution(
	nInputPlane, nOutputPlane,
	kW, kH, dW, dH, padW, padH)

	assert(padW);

	padH = padH or padW;

	local result;

	-- here is 2 options:
	if CUDNN_AVAILABLE then
		-- cudnn
		result = cudnn.SpatialConvolution(
			nInputPlane,
			nOutputPlane,
			kW, kH, dW, dH,
			padW, padH,
			1);
	else
		-- torch
		result = nn.SpatialConvolution(
			nInputPlane,
			nOutputPlane,
			kW, kH, dW, dH, padW, padH);
	end

	return result;
end


function makeSpatialMaxPooling(kW, kH, dW, dH, padW, padH)

	padH = padH or padW;

	assert(kW > 0);
	assert(kH > 0);
	assert(dW > 0);
	assert(dH > 0);
	assert(padW >= 0);
	assert(padH >= 0);

	if CUDNN_AVAILABLE then
		return cudnn.SpatialMaxPooling(
			kW, kH,
			dW, dH,
			padW, padH);
	else
		return nn.SpatialMaxPooling(
			kW, kH,
			dW, dH,
			padW, padH);
	end
end


function makeSpatialAveragePooling(kW, kH, dW, dH, padW, padH)

	padH = padH or padW;

	if CUDNN_AVAILABLE then
		return cudnn.SpatialAveragePooling(
			kW, kH,
			dW, dH,
			padW, padH  -- padW, padH
			);
	else
		return nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH);
	end

end







-- seq, elem1[, [elem2, [...]]
-- depth_concat, elem1[, [elem2, [...]]
-- maxpool, kw, kh, dw, dh, padW, padH
-- avgpool, kw, kh, dw, dh, padW, padH
-- activation, channels
-- conv in_c, out_c, kw, kh, dw, dh, padW, padH
function construct_structure(structure)
	local sname = structure[1];

	assert(torch.type(sname) == 'string');

	if sname == 'seq' or sname == 'depth_concat' then
		local r;
		if sname == 'seq' then
			r = nn.Sequential();
		else
			r = nn.DepthConcat(2);
		end

		for i = 2, #structure do
			r:add(construct_structure(structure[i]));
		end

		return r;
	elseif sname == 'wv' then
		return nn.View(1000000001);
	elseif sname == 'activation' then
		return makeActivationUnit(structure[2]);
	elseif sname == 'maxpool' then
		return makeSpatialMaxPooling(
			structure[2],
			structure[3],
			structure[4],
			structure[5],
			structure[6],
			structure[7]);
	elseif sname == 'avgpool' then
		return makeSpatialAveragePooling(
			structure[2],
			structure[3],
			structure[4],
			structure[5],
			structure[6],
			structure[7]);
	elseif sname == 'spatialdropout' then
		return nn.SpatialDropout(structure[2]);
	elseif sname == 'conv' then
		return makeSpatialConvolution(
			structure[2],
			structure[3],
			structure[4],
			structure[5],
			structure[6],
			structure[7],
			structure[8],
			structure[9]);
	else
		error('unnknown name "' .. sname '"');
	end
end


function inception_variants(mm)
	return {
		inception1 =
		{'depth_concat',
			{'seq',
				{'conv', 16*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm}},
			{'seq',
				{'conv', 16*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  5, 5,  1, 1,  2, 2}, {'activation', 8*mm}},
			{'seq',
				{'conv', 16*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  3, 3,  1, 1,  1, 1}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  3, 3,  1, 1,  1, 1}, {'activation', 8*mm}},
			{'seq',
				{'avgpool', 3, 3,  1, 1,  1, 1},
				{'conv', 16*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm}}},

		inception1ab =
		{'depth_concat',
			{'seq',
				{'conv', 32*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm}},
			{'seq',
				{'conv', 32*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  5, 5,  1, 1,  2, 2}, {'activation', 8*mm}},
			{'seq',
				{'conv', 32*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  3, 3,  1, 1,  1, 1}, {'activation', 8*mm},
				{'conv',  8*mm, 8*mm,  3, 3,  1, 1,  1, 1}, {'activation', 8*mm}},
			{'seq',
				{'avgpool', 3, 3,  1, 1,  1, 1},
				{'conv', 32*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm}}},

		inception2 =
		{'depth_concat',
			{'seq',
				{'conv', 32*mm, 16*mm,  3, 3,  2, 2,  1, 1}, {'activation', 16*mm}},
			{'seq',
				{'conv', 32*mm,  8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm,  16*mm,  3, 3,  1, 1,  1, 1}, {'activation', 16*mm},
				{'conv',  16*mm,  16*mm,  3, 3,  2, 2,  1, 1}, {'activation', 16*mm}},
			{'maxpool', 3, 3,  2, 2,  1, 1}},

		inception3ab =
		{'depth_concat',
			{'seq',
				{'conv', 64*mm, 16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm}},
			{'seq',
				{'conv', 64*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 16*mm,  5, 5,  1, 1,  2, 2}, {'activation', 16*mm}},
			{'seq',
				{'conv', 64*mm, 8*mm,  1, 1,  1, 1,  0, 0}, {'activation', 8*mm},
				{'conv',  8*mm, 16*mm,  3, 3,  1, 1,  1, 1}, {'activation', 16*mm},
				{'conv',  16*mm, 16*mm,  3, 3,  1, 1,  1, 1}, {'activation', 16*mm}},
			{'seq',
				{'avgpool', 3, 3,  1, 1,  1, 1},
				{'conv', 64*mm, 16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm}}},

		inception4 =
		{'depth_concat',
			{'seq',
				{'conv', 64*mm, 16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm},
				{'conv', 16*mm, 32*mm,  3, 3,  2, 2,  1, 1}, {'activation', 32*mm}},
			{'seq',
				{'conv', 64*mm,  16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm},
				{'conv',  16*mm,  16*mm,  1, 5,  1, 1,  0, 2}, {'activation', 16*mm},
				{'conv',  16*mm,  16*mm,  5, 1,  1, 1,  2, 0}, {'activation', 16*mm},
				{'conv',  16*mm,  32*mm,  3, 3,  2, 2,  1, 1}, {'activation', 32*mm}},
			{'maxpool', 3, 3,  2, 2,  1, 1}},

		inception5abc =
		{'depth_concat',
			{'seq',
				{'conv', 128*mm, 32*mm,  1, 1,  1, 1,  0, 0}, {'activation', 32*mm}},
			{'seq',
				{'conv', 128*mm, 16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm},
				{'depth_concat',
					{'seq', {'conv', 16*mm, 16*mm, 3, 1, 1, 1, 1, 0}, {'activation', 16*mm}},
					{'seq', {'conv', 16*mm, 16*mm, 1, 3, 1, 1, 0, 1}, {'activation', 16*mm}}}},
			{'seq',
				{'conv', 128*mm, 16*mm,  1, 1,  1, 1,  0, 0}, {'activation', 16*mm},
				{'conv',  16*mm, 16*mm,  3, 3,  1, 1,  1, 1}, {'activation', 16*mm},
				{'depth_concat',
					{'seq', {'conv',  16*mm, 16*mm,  3, 1,  1, 1,  1, 0}, {'activation', 16*mm}},
					{'seq', {'conv',  16*mm, 16*mm,  1, 3,  1, 1,  0, 1}, {'activation', 16*mm}}}},
			{'seq',
				{'avgpool', 3, 3,  1, 1,  1, 1},
				{'conv', 128*mm, 32*mm,  1, 1,  1, 1,  0, 0}, {'activation', 32*mm}}},
	};
end
