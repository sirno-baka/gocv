package cuda

/*
#include <stdlib.h>
#include "../core.h"
#include "core.h"
*/
import "C"
import (
	"image"
	"reflect"
	"unsafe"
)

func toRectangles(ret C.Rects) []image.Rectangle {
	cArray := ret.rects
	length := int(ret.length)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(cArray)),
		Len:  length,
		Cap:  length,
	}
	s := *(*[]C.Rect)(unsafe.Pointer(&hdr))

	rects := make([]image.Rectangle, length)
	for i, r := range s {
		rects[i] = image.Rect(int(r.x), int(r.y), int(r.x+r.width), int(r.y+r.height))
	}
	return rects
}

type KeyPoint struct {
	X, Y                  float64
	Size, Angle, Response float64
	Octave, ClassID       int
}

func getKeyPoints(ret C.KeyPoints) []KeyPoint {
	cArray := ret.keypoints
	length := int(ret.length)
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(cArray)),
		Len:  length,
		Cap:  length,
	}
	s := *(*[]C.KeyPoint)(unsafe.Pointer(&hdr))

	keys := make([]KeyPoint, length)
	for i, r := range s {
		keys[i] = KeyPoint{float64(r.x), float64(r.y), float64(r.size), float64(r.angle), float64(r.response),
			int(r.octave), int(r.classID)}
	}
	return keys
}
