#include "cudacodec.h"
#include <string>
#include <opencv2/cudacodec.hpp>

CudaVideoReader CudaVideoReader_Create(const char* filename, bool dropFrames) {
    cv::cudacodec::VideoReaderInitParams params;
    params.allowFrameDrop = dropFrames;
    return new cv::Ptr<cv::cudacodec::VideoReader>(cv::cudacodec::createVideoReader(std::string(filename), {}, params));
}

bool CudaVideoReader_nextFrame(CudaVideoReader b, GpuMat frame) {
    return (*b)->nextFrame(*frame);
}

void CudaVideoReader_Close(CudaVideoReader b) {
    delete b;
}

void CudaVideoReader_Set(CudaVideoReader b, int prop, double param) {
    (*b)->set(cv::cudacodec::VideoReaderProps(prop), param);
}

double CudaVideoReader_Get(CudaVideoReader b, int prop) {
    double value = 0;
    bool success = (*b)->get(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE, value);
    std::cout << success << std::endl;
    return value;
}

void copyFormatInfo(cv::cudacodec::FormatInfo cudaFormatInfo, FormatInfo& formatInfo) {
    formatInfo.codec = cudaFormatInfo.codec;
    formatInfo.chromaFormat = cudaFormatInfo.chromaFormat;
    formatInfo.width = cudaFormatInfo.width;
    formatInfo.height = cudaFormatInfo.height;
    formatInfo.fps = cudaFormatInfo.fps;

}

FormatInfo CudaVideoReader_Format(CudaVideoReader b) {
    cv::cudacodec::FormatInfo cudaFormatInfo = (*b)->format();
    FormatInfo formatInfo;

    copyFormatInfo(cudaFormatInfo, formatInfo);
    return formatInfo;
}

