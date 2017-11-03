#ifndef __TORCH_C_FUNCS__CFCOMMON_H__
#define __TORCH_C_FUNCS__CFCOMMON_H__

#include <luaT.h>
#include <TH.h>

#include <opencv2/opencv.hpp>

//  THFloatTensor
//  THByteTensor
template<typename THTensorType>
void assertContinuous(
	const THTensorType* const tensor)
{
	const int dim = tensor->nDimension;
	int stride = 1;
	for(int i = dim - 1; i >= 0; --i)
	{
		//RAssert(tensor->stride[i] == stride);

		stride *= tensor->size[i];
	}
}

inline
void write_tensor2cv_mat(
	THByteTensor const* const src,
	cv::Mat &dst)
{
	//RAssert(src);
	assertContinuous(src);
	//RAssert(src->nDimension == 3);

	const int channels = src->size[0];
	const int rows = src->size[1];
	const int cols = src->size[2];

	//RAssert(channels == 1 || channels == 3);

	dst.create(rows, cols, (channels == 1) ? CV_8UC1 : CV_8UC3);

	const uchar* const src_ptr = src->storage->data + src->storageOffset;

	for(int i = 0; i < rows; ++i)
	{
		uchar* const dst_ptr = dst.ptr<uchar>(i);
		for(int j = 0; j < cols; ++j)
			for(int c = 0; c < channels; ++c)
				dst_ptr[j * channels + c] =
					src_ptr[j + cols * (i  + rows * c)];
	}
}

inline
void write_cv_mat2tensor(
	const cv::Mat &src,
	THFloatTensor const* const dst)
{
	//RAssert(!src.empty());
	//RAssert(src.type() == CV_32FC1 || src.type() == CV_32FC3);

	//RAssert(dst);
	assertContinuous(dst);
	//RAssert(dst->nDimension == 3);

	const int channels = src.channels();
	const int rows = src.rows;
	const int cols = src.cols;

	//RAssert(dst->size[0] == channels);
	//RAssert(dst->size[1] == rows);
	//RAssert(dst->size[2] == cols);

	float* const dst_ptr = dst->storage->data + dst->storageOffset;
	for(int i = 0; i < rows; ++i)
	{
		const float* const src_ptr = src.ptr<float>(i);
		for(int j = 0; j < cols; ++j)
			for(int c = 0; c < channels; ++c)
				dst_ptr[j + cols * (i + rows * c)] =
					src_ptr[c + channels * j];
	}
}

inline
void write_vector_point2tensor(
        const std::vector<cv::Point2f> &src,
        THFloatTensor const* const dst)
{
    //RAssert(!src.empty());

    //RAssert(dst);
    assertContinuous(dst);
    //RAssert(dst->nDimension == 2);

    const int num_pts = src.size();

    //RAssert(dst->size[0] == 2);
    //RAssert(dst->size[1] == num_pts);

    float* const dst_ptr = dst->storage->data + dst->storageOffset;

    for(int i = 0; i < num_pts; ++i)
    {
        dst_ptr[i] = src.at(i).x;
        dst_ptr[i + num_pts] = src.at(i).y;
    }
}

inline
std::string tensor2string(
	THByteTensor const* const t)
{
	//RAssert(t);
	//RAssert(t->nDimension == 1);

	return (const char*)(t->storage->data + t->storageOffset);
}

#endif // __TORCH_C_FUNCS__CFCOMMON_H__