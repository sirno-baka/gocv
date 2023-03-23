package cuda

/*
#include <stdlib.h>
#include "cuda.h"
#include "xfeatures2d.h"
*/
import "C"
import (
	"reflect"
	"unsafe"
)

// SURF is a wrapper around the cv::cuda::SURF_CUDA.
type SURF struct {
	// C.ORB
	p unsafe.Pointer
}

// NewSURF returns a new SURF
func NewSURF() SURF {
	return SURF{p: unsafe.Pointer(C.CudaSURF_Create())}
}

func (o *SURF) DetectAndCompute(src GpuMat, mask GpuMat) ([]KeyPoint, GpuMat) {
	desc := NewGpuMat()
	ret := C.CudaSURF_DetectAndCompute((C.CudaSURF)(o.p), src.p, mask.p, desc.p)
	return getKeyPoints(ret), desc
}

type DMatch struct {
	QueryIdx int
	TrainIdx int
	ImgIdx   int
	Distance float64
}

func getMultiDMatches(ret C.MultiDMatches) [][]DMatch {
	cArray := ret.dmatches
	length := int(ret.length)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(cArray)),
		Len:  length,
		Cap:  length,
	}
	s := *(*[]C.DMatches)(unsafe.Pointer(&hdr))

	keys := make([][]DMatch, length)
	for i := range s {
		keys[i] = getDMatches(C.CudaMultiDMatches_get(ret, C.int(i)))
	}
	return keys
}

func getDMatches(ret C.DMatches) []DMatch {
	cArray := ret.dmatches
	length := int(ret.length)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(cArray)),
		Len:  length,
		Cap:  length,
	}
	s := *(*[]C.DMatch)(unsafe.Pointer(&hdr))

	keys := make([]DMatch, length)
	for i, r := range s {
		keys[i] = DMatch{int(r.queryIdx), int(r.trainIdx), int(r.imgIdx),
			float64(r.distance)}
	}
	return keys
}

// BFMatcher is a wrapper around the the cv::cuda::DescriptionMatcher algorithm
type BFMatcher struct {
	// C.BFMatcher
	p unsafe.Pointer
}

// NewBFMatcher returns a new BFMatcher
func NewBFMatcher() BFMatcher {
	return BFMatcher{p: unsafe.Pointer(C.CudaBFMatcher_Create())}
}

// Close BFMatcher
func (b *BFMatcher) Close() error {
	C.CudaBFMatcher_Close((C.CudaBFMatcher)(b.p))
	b.p = nil
	return nil
}

// KnnMatch Finds the k best matches for each descriptor from a query set.
func (b *BFMatcher) KnnMatch(query, train GpuMat, k int) [][]DMatch {
	ret := C.CudaBFMatcher_KnnMatch((C.CudaBFMatcher)(b.p), query.p, train.p, C.int(k))
	defer C.CudaMultiDMatches_Close(ret)

	return getMultiDMatches(ret)
}
