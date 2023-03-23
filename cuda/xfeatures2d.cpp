#include "xfeatures2d.h"

CudaSURF CudaSURF_Create(double threshold ) {
    return new cv::Ptr<cv::cuda::SURF_CUDA>(cv::cuda::SURF_CUDA::create(threshold));
}

KeyPoints CudaSURF_DetectAndCompute(CudaSURF surf, GpuMat src, GpuMat mask, GpuMat desc){
    cv::cuda::GpuMat keypoints;

    (*(*surf))(*src, *mask, keypoints, *desc);

    std::vector<cv::KeyPoint> detected;
    (*(*surf)).downloadKeypoints(keypoints, detected);
    
    KeyPoint* kps = new KeyPoint[detected.size()];
    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }
    KeyPoints ret = {kps, (int)detected.size()};
    delete[] ret.keypoints;
    return ret;
}





CudaBFMatcher CudaBFMatcher_Create() {
    return new cv::Ptr<cv::cuda::DescriptorMatcher>(cv::cuda::DescriptorMatcher::createBFMatcher());
}

void CudaBFMatcher_Close(CudaBFMatcher b) {
    delete b;
}

struct MultiDMatches CudaBFMatcher_KnnMatch(CudaBFMatcher b, GpuMat query, GpuMat train, int k) {
    std::vector< std::vector<cv::DMatch> > matches;
    (*b)->knnMatch(*query, *train, matches, k);

    DMatches *dms = new DMatches[matches.size()];
    for (size_t i = 0; i < matches.size(); ++i) {
        DMatch *dmatches = new DMatch[matches[i].size()];
        for (size_t j = 0; j < matches[i].size(); ++j) {
            DMatch dmatch = {matches[i][j].queryIdx, matches[i][j].trainIdx, matches[i][j].imgIdx,
                             matches[i][j].distance};
            dmatches[j] = dmatch;
        }
        dms[i] = {dmatches, (int) matches[i].size()};
    }
    MultiDMatches ret = {dms, (int) matches.size()};
    return ret;
}