#include "vpi_orb.h"

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/ImageFlip.h>
#include <vpi/algo/ORB.h>

VPIImage imgInput     = NULL;
VPIImage imgGrayScale = NULL;

VPIPyramid pyrInput   = NULL;
VPIArray keypoints    = NULL;
VPIArray descriptors  = NULL;
VPIPayload orbPayload = NULL;
VPIStream stream      = NULL;
VPIBackend backend;
VPIBorderExtension be;
VPIORBParams orbParams;

 #define CHECK_STATUS(STMT)                                    \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);




static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypointF32 *kpts, VPIBriefDescriptor *outDescriptors, int numKeypoints)
 {
     cv::Mat out;
     img.convertTo(out, CV_8UC1);
     cvtColor(out, out, cv::COLOR_GRAY2BGR);

     if (numKeypoints == 0)
     {
         return out;
     }

     for (int i = 0; i < numKeypoints; ++i)
       {
     auto de=outDescriptors[i];
     cv::Scalar col(255,0,0);
     for(int j=0; j<32;j++){
       if(de.data[j]>0) { col=cv::Scalar(0,255, 0); break;}
     }
     circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, col, -1);
       }
     return out;
 }

void initOrb(const char* strBackend){
         std::cout << strBackend << std::endl;
         if ( strcmp(strBackend, "cpu") == 0)
         {
             backend = VPI_BACKEND_CPU;
         }
         else if (strcmp(strBackend, "cuda") == 0)
         {
             backend = VPI_BACKEND_CUDA;
         }
         else
         {
             throw std::runtime_error("Backend  not recognized, it must be either cpu or cuda.");
         }

         // =================================
         // Allocate all VPI resources needed

         // Create the stream where processing will happen
         CHECK_STATUS(vpiStreamCreate(0, &stream));

         // Define the algorithm parameters.
         CHECK_STATUS(vpiInitORBParams(&orbParams));
              //default params
              //params.fastParams.circleRadius       = 3;
              //params.fastParams.arcLength          = 9;
              //params.fastParams.intensityThreshold = 20;
              //params.fastParams.nonMaxSuppression  = 1;
              //params.maxFeatures                   = 100;
              //params.pyramidLevels                 = 4;
              //params.enableRBRIEF                  = true;
              //params.scoreType                     = VPI_CORNER_SCORE_HARRIS;


         orbParams.scoreType=VPI_CORNER_SCORE_HARRIS;
//         orbParams.scoreType=VPI_CORNER_SCORE_FAST;
         orbParams.maxFeatures=10000;
         orbParams.pyramidLevels=3;

         be= VPI_BORDER_ZERO; //VPI_BORDER_CLAMP; //VPI_BORDER_ZERO
}

KeyPoints OrbDetect(Mat cvImage, Mat outDescriptors) {
    // We now wrap the loaded image into a VPIImage object to be used by VPI.
    // VPI won't make a copy of it, so the original
    // image must be in scope at all times.
    CHECK_STATUS(vpiImageCreateWrapperOpenCVMat(*cvImage, 0, &imgInput));
    CHECK_STATUS(vpiImageCreate(cvImage->cols, cvImage->rows, VPI_IMAGE_FORMAT_U8, 0, &imgGrayScale));

    // Create the output keypoint array.
    CHECK_STATUS( vpiArrayCreate(orbParams.maxFeatures, VPI_ARRAY_TYPE_KEYPOINT_F32, backend | VPI_BACKEND_CPU, &keypoints));

    // Create the output descriptors array.
    CHECK_STATUS(vpiArrayCreate(orbParams.maxFeatures, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR, backend | VPI_BACKEND_CPU, &descriptors));

    // Create the payload for ORB Feature Detector algorithm
    CHECK_STATUS( vpiCreateORBFeatureDetector(backend, 20000, &orbPayload));

    // ================
    // Processing stage

    // First convert input to grayscale
    CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backend, imgInput, imgGrayScale, NULL));

    // Then, create the Gaussian Pyramid for the image and wait for the execution to finish
    CHECK_STATUS(vpiPyramidCreate(cvImage->cols, cvImage->rows, VPI_IMAGE_FORMAT_U8, orbParams.pyramidLevels, 0.5, backend, &pyrInput));
    CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgGrayScale, pyrInput, be));

    // Then get ORB features and wait for the execution to finish
    CHECK_STATUS(vpiSubmitORBFeatureDetector(stream, backend, orbPayload, pyrInput, keypoints, descriptors,  &orbParams, be));

    CHECK_STATUS(vpiStreamSync(stream));

    // =======================================
    // Output processing and saving it to disk

    // Lock output keypoints and scores to retrieve its data on cpu memory
    VPIArrayData outKeypointsData;
    VPIArrayData outDescriptorsData;
    CHECK_STATUS(vpiArrayLockData(keypoints, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outKeypointsData));
    CHECK_STATUS(vpiArrayLockData(descriptors, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outDescriptorsData));
    cv::Mat des( *outDescriptorsData.buffer.aos.sizePointer / 32, 32, CV_8UC1, outDescriptorsData.buffer.aos.data);
//    std::cout << des.cols << std::endl;
    (*outDescriptors) = des.clone();
    des.release();
//    VPIKeypointF32 *outKeypoints2 = (VPIKeypointF32 *)outKeypointsData.buffer.aos.data;
//    VPIBriefDescriptor *outDescriptors2 = (VPIBriefDescriptor *) outDescriptorsData.buffer.aos.data;
//    VPIImageData imgData;
//    CHECK_STATUS(vpiImageLockData(imgGrayScale, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));
//    cv::Mat img;
//   CHECK_STATUS(vpiImageDataExportOpenCVMat(imgData, &img));
//   cv::Mat outImage = DrawKeypoints(img, outKeypoints2, outDescriptors2, *outKeypointsData.buffer.aos.sizePointer);
//   cv::imshow("Display window", outImage);
//   cv::waitKey(22);

     int nkp = *outKeypointsData.buffer.aos.sizePointer;
     VPIKeypointF32 *outKeypoints = (VPIKeypointF32 *)outKeypointsData.buffer.aos.data;
     KeyPoint* kps = new KeyPoint[nkp];

     for (int i=0; i<nkp;i++) {
          KeyPoint k = {(outKeypoints[i]).x, (outKeypoints[i]).y, 0, 0, 0, 0, 0};
             kps[i] = k;
     }
    KeyPoints ret = {kps, nkp};
    return ret;
}
void clean() {
     //CHECK_STATUS(vpiImageUnlock(imgGrayScale));
     CHECK_STATUS(vpiArrayUnlock(keypoints));
     CHECK_STATUS(vpiArrayUnlock(descriptors));
     vpiPyramidDestroy(pyrInput);
     vpiStreamSync(stream);
     vpiImageDestroy(imgInput);
     vpiImageDestroy(imgGrayScale);
     vpiArrayDestroy(keypoints);
     vpiArrayDestroy(descriptors);
     vpiPayloadDestroy(orbPayload);
 }
void destroy(){
    clean();
    vpiStreamDestroy(stream);
}

KeyPoint * returnKeypoints(std::pair<VPIArrayData, VPIArrayData> result) {
    KeyPoint *points = (KeyPoint *)result.first.buffer.aos.data;
    return points;
}

int mainTest()
 {
     // VPI objects that will be used
     int retval = 0;

     try
     {
         initOrb("cuda");
         // =====================
         // Load the input image
         cv::Mat cvImage;
         cv::Mat des;
         cv::namedWindow("Display window");
         cv::VideoCapture cap(1);
         cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
         cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
//         cap.set(cv::CAP_PROP_FPS, 50);
//         cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
         if (!cap.isOpened()) {
         std::cout << "cannot open camera";
         }
         while (true) {
         int64 start = cv::getTickCount();
         cap >> cvImage;
         if (cvImage.empty())
         {
             throw std::runtime_error("Can't open image");
         }
         //std::pair <VPIArrayData, VPIArrayData> result;
         OrbDetect(&cvImage, &des);

         //VPIKeypointF32 *outKeypoints = (VPIKeypointF32 *)result.first.buffer.aos.data ;
         //VPIBriefDescriptor *outDescriptors= (VPIBriefDescriptor *) result.second.buffer.aos.data ;
         //int nkp = *result.first.buffer.aos.sizePointer;
         //int nde = *result.second.buffer.aos.sizePointer;


         /*
             printf("%d keypoints found\n", nkp);
             printf("%d descriptors found\n", nde);

              printf("%d keypoints found\n", nkp);
              for(int i=0; i<nkp;i++)
              {
                printf("%d %f %f\n",i, (outKeypoints[i]).x,(outKeypoints[i]).y);
              }

              printf("%d descriptors found\n", nde);
              for(int i=0; i<nde;i++)
                 {
                printf("%d: ",i);
                for(int j=0; j<VPI_BRIEF_DESCRIPTOR_ARRAY_LENGTH; j++)
                {
                  printf("%hhu ", (outDescriptors[i]).data[j]);
                }
                printf("\n");
              }
          */
         // Done handling outputs, don't forget to unlock them.
         double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
         std::cout << "FPS : " << fps << std::endl;
         clean();
     }
     }
     catch (std::exception &e)
     {
         std::cerr << e.what() << std::endl;
         retval = 1;
     }
     destroy();
     return  retval;
 }
