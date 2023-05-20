#include "cudacodec.h"


CudaVideoReader CudaVideoReader_Create(const char* filename) {
    return new cv::Ptr<cv::cudacodec::VideoReader>(cv::cudacodec::createVideoReader(std::string(filename)));
}

bool CudaVideoReader_nextFrame(CudaVideoReader b, GpuMat frame) {
    return (*b)->nextFrame(*frame);
}

void CudaVideoReader_Close(CudaVideoReader b) {
    delete b;
}

