#ifndef VPI_ORB_MAIN_H
#define VPI_ORB_MAIN_H
#include "core.h"


#ifdef __cplusplus
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <vpi/OpenCVInterop.hpp>
#include <bitset>
#include <cstdio>
#include <cstring> // for memset
#include <iostream>
#include <sstream>
#include<tuple>
#include <string>


extern "C" {
#endif

#ifdef __cplusplus
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>
static Mat DrawKeypoints(Mat img, VPIKeypointF32 *kpts, VPIBriefDescriptor *outDescriptors, int numKeypoints);
#endif


int mainTest();

void initOrb(const char* strBackend, int maxFeatures, int intensityThreshold);
KeyPoints OrbDetect(Mat cvImage, Mat descriptions);
void clean();
void destroy();

#ifdef __cplusplus
}
#endif



#endif