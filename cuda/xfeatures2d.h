#ifndef _OPENCV3_CUDAFEATURES2D_H_
#define _OPENCV3_CUDAFEATURES2D_H_

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
extern "C" {
#endif

#include "../core.h"
#include "cuda.h"

#ifdef __cplusplus
typedef cv::Ptr<cv::cuda::SURF_CUDA>* CudaSURF;
typedef cv::Ptr<cv::cuda::DescriptorMatcher>* CudaBFMatcher;

#else
typedef void* CudaSURF;
typedef void* CudaBFMatcher;
#endif

CudaSURF CudaSURF_Create();
KeyPoints CudaSURF_DetectAndCompute(CudaSURF o, GpuMat src, GpuMat mask, GpuMat desc);

CudaBFMatcher CudaBFMatcher_Create();
void CudaBFMatcher_Close(CudaBFMatcher b);
struct MultiDMatches CudaBFMatcher_KnnMatch(CudaBFMatcher b, GpuMat query, GpuMat train, int k);

#ifdef __cplusplus
}
#endif

#endif // _OPENCV3_CUDAFEATURES2D_H_