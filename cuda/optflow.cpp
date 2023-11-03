#include "optflow.h"

CudaSparsePyrLKOpticalFlow CudaSparsePyrLKOpticalFlow_Create() {
    return new cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>(cv::cuda::SparsePyrLKOpticalFlow::create());
}

void CudaSparsePyrLKOpticalFlow_Calc(CudaSparsePyrLKOpticalFlow p, GpuMat prevImg, GpuMat nextImg, GpuMat prevPts, GpuMat nextPts, GpuMat status){
    (*p)->calc(*prevImg,*nextImg,*prevPts,*nextPts,*status);
}


CudaFarnebackOpticalFlow CudaFarnebackOpticalFlow_Create(int numLevels) {
    return new cv::Ptr<cv::cuda::FarnebackOpticalFlow>(cv::cuda::FarnebackOpticalFlow::create(numLevels));
}

void CudaFarnebackOpticalFlow_Calc(CudaFarnebackOpticalFlow p, GpuMat prevImg, GpuMat nextImg, GpuMat flow){
    (*p)->calc(*prevImg,*nextImg, *flow);
}