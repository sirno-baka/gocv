#ifndef _OPENCV3_CUDACODEC_H_
#define _OPENCV3_CUDACODEC_H_

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
extern "C" {
#endif

#include "../core.h"
#include "cuda.h"

#ifdef __cplusplus
typedef cv::Ptr<cv::cudacodec::VideoReader>* CudaVideoReader;
#else
typedef void* CudaVideoReader;
#endif
typedef struct FormatInfo {
    int codec;
    int chromaFormat;
    int width;
    int height;
    int bitDepth;
    int fps;
} FormatInfo;


CudaVideoReader CudaVideoReader_Create(const char* filename, bool dropFrames);

bool CudaVideoReader_nextFrame(CudaVideoReader b, GpuMat frame);
void CudaVideoReader_Close(CudaVideoReader b);
void CudaVideoReader_Set(CudaVideoReader v, int prop, double param);
double CudaVideoReader_Get(CudaVideoReader v, int prop);
FormatInfo CudaVideoReader_Format(CudaVideoReader v);

#ifdef __cplusplus
}
#endif

#endif