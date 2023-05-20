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

type CudaVideoReader struct {
	p unsafe.Pointer
}

func NewCudaVideoReader(filename string) CudaVideoReader {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	return CudaVideoReader{p: unsafe.Pointer(C.CudaVideoReader_Create(cFilename))}
}

func (o *CudaVideoReader) NextFrame(frame GpuMat) bool {
	ret := C.CudaVideoReader_nextFrame((C.CudaVideoReader)(o.p), frame.p)
	return bool(ret)
}

func (o *CudaVideoReader) Close() error {
	C.CudaVideoReader_Close((C.CudaVideoReader)(o.p))
	o.p = nil
	return nil
}
