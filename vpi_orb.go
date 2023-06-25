package gocv

/*
#include <stdlib.h>
#include "vpi_orb.h"
#cgo LDFLAGS: -L /usr/lib -lnvvpi

*/
import "C"
import (
	"unsafe"
)

func OrbInit(strBackend string, maxFeatures int, intensityThreshold int) {

	parameter := C.CString(strBackend)
	defer C.free(unsafe.Pointer(parameter))
	C.initOrb(parameter, C.int(maxFeatures), C.int(intensityThreshold))

}

func OrbClean() {
	C.clean()
}

func OrbDestroy() {
	C.destroy()
}

func OrbDetect(mat Mat) ([]KeyPoint, Mat) {
	des := NewMat()
	s := C.OrbDetect(mat.p, des.p)
	keys := getKeyPoints(s)
	return keys, des
}

//
//func OrbMainTest() {
//	C.mainTest()
//}
//
//func OrbTestIntegration() {
//	// set to use a video capture device 0
//	deviceID := 1
//	OrbInit("cuda", 10000, 5)
//	// open webcam
//	webcam, err := VideoCaptureDeviceWithAPI(deviceID, VideoCaptureV4L2)
//	if err != nil {
//		fmt.Println(err)
//		return
//	}
//	webcam.Set(VideoCaptureFPS, 60)
//	webcam.Set(VideoCaptureFrameWidth, 1920)
//	webcam.Set(VideoCaptureFrameHeight, 1080)
//	webcam.Set(VideoCaptureFOURCC, webcam.ToCodec("MJPG"))
//	defer webcam.Close()
//	// open display window
//	window := NewWindow("Face Detect")
//	//window2 := NewWindow("Face 2")
//	defer window.Close()
//	// prepare image matrix
//	img := NewMat()
//
//	orb := NewORB()
//	mask := NewMat()
//	fmt.Printf("start reading camera device: %v\n", deviceID)
//	for {
//		if ok := webcam.Read(&img); !ok {
//			fmt.Printf("cannot read device %v\n", deviceID)
//			return
//		}
//		if img.Empty() {
//			continue
//		}
//		kp, d := OrbDetect(img)
//		kp2, d2 := orb.DetectAndCompute(img, mask)
//		for _, p := range kp {
//			Circle(&img, image.Point{int(p.X), int(p.Y)}, 1, color.RGBA{10, 222, 10, 222}, 2)
//		}
//		for _, p := range kp2 {
//			Circle(&img, image.Point{int(p.X), int(p.Y)}, 1, color.RGBA{222, 10, 10, 222}, 2)
//		}
//		OrbClean()
//		fmt.Println("vpi", d.Cols(), d.Rows())
//		fmt.Println("orb", d2.Cols(), d2.Rows())
//		println()
//		window.IMShow(img)
//		//window2.WaitKey(2)
//		window.WaitKey(2)
//	}
//	OrbDestroy()
//}
