#ifndef _OPENCV_CUDAOPTFLOW_HPP_
#define _OPENCV_CUDAOPTFLOW_HPP_

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

extern "C" {
#endif

#include "../core.h"
#include "cuda.h"

#ifdef __cplusplus
typedef cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>* CudaSparsePyrLKOpticalFlow;
typedef cv::Ptr<cv::cuda::FarnebackOpticalFlow>* CudaFarnebackOpticalFlow;
#else
typedef void* CudaSparsePyrLKOpticalFlow;
typedef void* CudaFarnebackOpticalFlow;
#endif

CudaSparsePyrLKOpticalFlow CudaSparsePyrLKOpticalFlow_Create();
void CudaSparsePyrLKOpticalFlow_Calc(CudaSparsePyrLKOpticalFlow p, GpuMat prevImg, GpuMat nextImg, GpuMat prevPts, GpuMat nextPts, GpuMat status);

CudaFarnebackOpticalFlow CudaFarnebackOpticalFlow_Create(int numLevels);
void CudaFarnebackOpticalFlow_Calc(CudaFarnebackOpticalFlow p, GpuMat prevImg, GpuMat nextImg, GpuMat flow);


#ifdef __cplusplus
}
#endif

#endif // _OPENCV_CUDAOPTFLOW_HPP_