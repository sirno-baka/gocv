package cuda

/*
#include <stdlib.h>
#include "cuda.h"
#include "cudacodec.h"
*/
import "C"
import (
	"unsafe"
)

type Codec int

const (
	CodecUncompressed Codec = iota
	CodecMJPEG
	CodecMPEG1
	CodecMPEG2
	CodecMPEG4
	CodecWMV1
	CodecWMV2
	CodecH263
	CodecH264
	CodecH265
	CodecVP8
	CodecVP9
	CodecHEVC
)

type ChromaFormat int

const (
	ChromaFormatMonochrome ChromaFormat = iota
	ChromaFormat420
	ChromaFormat422
	ChromaFo
)

type FormatInfo struct {
	Codec        Codec        // кодек видео
	ChromaFormat ChromaFormat // формат цветности
	Width        int          // ширина кадра
	Height       int          // высота кадра
	BitDepth     int          // глубина цвета
	FrameRate    int          // частота кадров
}

type CudaVideoReader struct {
	p unsafe.Pointer
}

func NewCudaVideoReader(filename string, dropFrames bool) CudaVideoReader {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	return CudaVideoReader{p: unsafe.Pointer(C.CudaVideoReader_Create(cFilename, C.bool(dropFrames)))}
}

func (o *CudaVideoReader) NextFrame(frame GpuMat) bool {
	ret := C.CudaVideoReader_nextFrame((C.CudaVideoReader)(o.p), frame.p)
	return bool(ret)
}

// Get parameter with property (=key).
func (v CudaVideoReader) Format() FormatInfo {
	cvFormat := C.CudaVideoReader_Format((C.CudaVideoReader)(v.p))
	return FormatInfo{
		Codec:        Codec(cvFormat.codec),
		ChromaFormat: ChromaFormat(cvFormat.chromaFormat),
		Width:        int(cvFormat.width),
		Height:       int(cvFormat.height),
		BitDepth:     int(cvFormat.bitDepth),
		FrameRate:    int(cvFormat.fps),
	}
}

func (o *CudaVideoReader) Close() error {
	C.CudaVideoReader_Close((C.CudaVideoReader)(o.p))
	o.p = nil
	return nil
}
