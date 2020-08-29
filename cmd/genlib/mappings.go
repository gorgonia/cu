package main

var empty struct{}
var ignoredFunctions = map[string]struct{}{
	/* Manually written */

	"cuGetErrorString":   empty, // ignored because Go can provide better contexts for the errors
	"cuGetErrorName":     empty, // ignored because Go can provide better contexts for the errors
	"cuInit":             empty,
	"cuDriverGetVersion": empty,
	"cuDeviceGetName":    empty, // wat?
	"cuDeviceGetUuid":    empty,

	// context stuff
	"cuCtxCreate":                 empty,
	"cuCtxDestroy":                empty,
	"cuDevicePrimaryCtxRetain":    empty,
	"cuCtxResetPersistingL2Cache": empty,

	// pointer/memory/unified addressing stuff
	"cuPointerGetAttribute":   empty,
	"cuMemPrefetchAsync":      empty,
	"cuMemAdvise":             empty,
	"cuMemRangeGetAttribute":  empty,
	"cuMemRangeGetAttributes": empty,
	"cuPointerSetAttribute":   empty,
	"cuPointerGetAttributes":  empty,
	"cuMemHostRegister":       empty,
	"cuMemHostUnregister":     empty,
	"cuMemGetAddressRange":    empty,

	// dealing with voids and strings...
	"cuLaunchKernel":                       empty,
	"cuLaunchCooperativeKernel":            empty, // TODO
	"cuLaunchCooperativeKernelMultiDevice": empty, // TODO - possibly never (no bandwidth)
	"cuLaunchHostFunc":                     empty, // TODO - possibly never, given the intricacies of calling Go functions in C.
	"cuModuleLoad":                         empty, // dealing with strings
	"cuModuleLoadData":                     empty, // dealing with strings
	"cuModuleGetFunction":                  empty, // dealing with strings
	"cuModuleGetGlobal":                    empty, // dealing with strings

	// event stuff
	"cuEventCreate":  empty,
	"cuEventDestroy": empty,

	// stream stuff
	"cuStreamCreate":             empty,
	"cuStreamCreateWithPriority": empty,
	"cuStreamDestroy":            empty,

	// arrays
	"cuArrayCreate":   empty,
	"cuArray3DCreate": empty,

	// occupany stuff
	"cuOccupancyMaxActiveBlocksPerMultiprocessor":          empty,
	"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": empty,

	// graph stuff
	"cuGraphCreate":  empty,
	"cuGraphDestroy": empty,
	"cuGraphClone":   empty,

	/* SUPPORT PLANNED BUT NOT YET DONE */
	// memory stuff
	"cuMemAllocHost":            empty, // use C.malloc
	"cuMemHostAlloc":            empty, // use C.malloc, be sure to allocate page-aligned ones
	"cuMemHostGetDevicePointer": empty,
	"cuMemHostGetFlags":         empty,

	// module/JIT stuff
	"cuModuleLoadDataEx":    empty,
	"cuModuleLoadFatBinary": empty,
	"cuModuleGetTexRef":     empty,
	"cuModuleGetSurfRef":    empty,
	"cuLinkCreate":          empty,
	"cuLinkAddData":         empty,
	"cuLinkAddFile":         empty,
	"cuLinkComplete":        empty,
	"cuLinkDestroy":         empty,

	/* Not planning to support anytime soon as these require extra attention */

	// NvSciSync
	"cuDeviceGetNvSciSyncAttributes": empty, // I have no idea what NvSciSync is.

	// Stream Batching
	"cuStreamBatchMemOp": empty,

	// hardware interaction
	"cuDeviceGetByPCIBusId": empty,
	"cuDeviceGetPCIBusId":   empty,

	// MPI stuff
	"cuIpcGetEventHandle":         empty,
	"cuIpcOpenEventHandle":        empty,
	"cuIpcGetMemHandle":           empty,
	"cuIpcOpenMemHandle":          empty,
	"cuIpcCloseMemHandle":         empty,
	"cuMipmappedArrayCreate":      empty,
	"cuMipmappedArrayGetLevel":    empty,
	"cuMipmappedArrayDestroy":     empty,
	"cuTexRefGetMipmapFilterMode": empty,
	"cuTexRefGetMipmapLevelBias":  empty,
	"cuTexRefGetMipmapLevelClamp": empty,
	"cuTexRefSetMipmappedArray":   empty,
	"cuTexRefGetMipmappedArray":   empty,

	// Function stuff
	"cuOccupancyMaxPotentialBlockSize":          empty,
	"cuOccupancyMaxPotentialBlockSizeWithFlags": empty,
	"cuStreamAddCallback":                       empty, // really only valid for C API calls in C programs

	// Graphics Interop
	"cuGraphicsUnregisterResource":              empty,
	"cuGraphicsSubResourceGetMappedArray":       empty,
	"cuGraphicsResourceGetMappedMipmappedA":     empty,
	"cuGraphicsResourceGetMappedPointer":        empty,
	"cuGraphicsResourceSetMapFlags":             empty,
	"cuGraphicsMapResources":                    empty,
	"cuGraphicsUnmapResources":                  empty,
	"cuGraphicsResourceGetMappedMipmappedArray": empty,

	// texture and surface object
	"cuTexObjectCreate":              empty,
	"cuTexObjectDestroy":             empty,
	"cuTexObjectGetResourceDesc":     empty,
	"cuTexObjectGetTextureDesc":      empty,
	"cuTexObjectGetResourceViewDesc": empty,
	"cuSurfObjectCreate":             empty,
	"cuSurfObjectDestroy":            empty,
	"cuSurfObjectGetResourceDesc":    empty,

	// I have no clue what this is
	"cuGetExportTable": empty,
	"cuFuncGetModule":  empty, // NOT IN DOCS

	// Deprecated from CUDA 8 API:
	"cuDeviceGetProperties":     empty,
	"cuDeviceComputeCapability": empty,
	"cuCtxAttach":               empty,
	"cuCtxDetach":               empty,
	"cuFuncSetBlockShape":       empty,
	"cuFuncSetSharedSize":       empty,
	"cuParamSetSize":            empty,
	"cuParamSeti":               empty,
	"cuParamSetf":               empty,
	"cuParamSetv":               empty,
	"cuLaunch":                  empty,
	"cuLaunchGrid":              empty,
	"cuLaunchGridAsync":         empty,
	"cuParamSetTexRef":          empty,
	"cuTexRefCreate":            empty,
	"cuTexRefDestroy":           empty,

	// virtual memory stuff - TODO because I don't have time right now
	"cuMemAddressFree":                       empty,
	"cuMemAddressReserve":                    empty,
	"cuMemCreate":                            empty,
	"cuMemExportToShareableHandle":           empty,
	"cuMemGetAccess":                         empty,
	"cuMemGetAllocationGranularity":          empty,
	"cuMemGetAllocationPropertiesFromHandle": empty,
	"cuMemImportFromShareableHandle":         empty,
	"cuMemMap":                               empty,
	"cuMemRelease":                           empty,
	"cuMemRetainAllocationHandle":            empty,
	"cuMemSetAccess":                         empty,
	"cuMemUnmap":                             empty,

	// External resource interop - UNSUPPORTED SO FAR
	"cuDestroyExternalMemory":                 empty,
	"cuDestroyExternalSemaphore":              empty,
	"cuExternalMemoryGetMappedBuffer":         empty,
	"cuExternalMemoryGetMappedMipmappedArray": empty,
	"cuImportExternalMemory":                  empty,
	"cuImportExternalSemaphore":               empty,
	"cuSignalExternalSemaphoresAsync":         empty,
	"cuWaitExternalSemaphoresAsync":           empty,

	// TEMP TODO
	"cuGraphAddChildGraphNode":                empty,
	"cuGraphAddDependencies":                  empty,
	"cuGraphAddEmptyNode":                     empty,
	"cuGraphAddHostNode":                      empty,
	"cuGraphAddKernelNode":                    empty,
	"cuGraphAddMemcpyNode":                    empty,
	"cuGraphAddMemsetNode":                    empty,
	"cuGraphChildGraphNodeGetGraph":           empty,
	"cuGraphDestroyNode":                      empty,
	"cuGraphExecDestroy":                      empty,
	"cuGraphExecHostNodeSetParams":            empty,
	"cuGraphExecKernelNodeSetParams":          empty,
	"cuGraphExecMemcpyNodeSetParams":          empty,
	"cuGraphExecMemsetNodeSetParams":          empty,
	"cuGraphExecUpdate":                       empty,
	"cuGraphGetEdges":                         empty,
	"cuGraphGetNodes":                         empty,
	"cuGraphGetRootNodes":                     empty,
	"cuGraphHostNodeGetParams":                empty,
	"cuGraphHostNodeSetParams":                empty,
	"cuGraphInstantiate":                      empty,
	"cuGraphKernelNodeCopyAttributes":         empty,
	"cuGraphKernelNodeGetAttribute":           empty,
	"cuGraphKernelNodeGetParams":              empty,
	"cuGraphKernelNodeSetAttribute":           empty,
	"cuGraphKernelNodeSetParams":              empty,
	"cuGraphLaunch":                           empty,
	"cuGraphMemcpyNodeGetParams":              empty,
	"cuGraphMemcpyNodeSetParams":              empty,
	"cuGraphMemsetNodeGetParams":              empty,
	"cuGraphMemsetNodeSetParams":              empty,
	"cuGraphNodeFindInClone":                  empty,
	"cuGraphNodeGetDependencies":              empty,
	"cuGraphNodeGetDependentNodes":            empty,
	"cuGraphNodeGetType":                      empty,
	"cuGraphRemoveDependencies":               empty,
	"cuOccupancyAvailableDynamicSMemPerBlock": empty,
}

var fnNameMap = map[string]string{
	"cuDeviceGet":                "GetDevice",
	"cuDeviceGetCount":           "NumDevices",
	"cuDeviceTotalMem":           "Device TotalMem",
	"cuDeviceGetAttribute":       "Device Attribute",
	"cuDevicePrimaryCtxRetain":   "Device RetainPrimaryCtx",
	"cuDevicePrimaryCtxRelease":  "Device ReleasePrimaryCtx",
	"cuDevicePrimaryCtxSetFlags": "Device SetPrimaryCtxFlags",
	"cuDevicePrimaryCtxGetState": "Device PrimaryCtxState",
	"cuDevicePrimaryCtxReset":    "Device ResetPrimaryCtx",

	"cuCtxCreate":                 "Device MakeContext",
	"cuCtxDestroy":                "DestroyContext",
	"cuCtxPushCurrent":            "PushCurrentCtx",
	"cuCtxPopCurrent":             "PopCurrentCtx",
	"cuCtxSetCurrent":             "SetCurrentContext",
	"cuCtxGetCurrent":             "CurrentContext",
	"cuCtxGetDevice":              "CurrentDevice",
	"cuCtxGetFlags":               "CurrentFlags",
	"cuCtxSynchronize":            "Synchronize",
	"cuCtxSetLimit":               "SetLimit",
	"cuCtxGetLimit":               "Limits",
	"cuCtxGetCacheConfig":         "CurrentCacheConfig",
	"cuCtxSetCacheConfig":         "SetCurrentCacheConfig",
	"cuCtxGetSharedMemConfig":     "SharedMemConfig",
	"cuCtxSetSharedMemConfig":     "SetSharedMemConfig",
	"cuCtxGetApiVersion":          "CUContext APIVersion",
	"cuCtxGetStreamPriorityRange": "StreamPriorityRange",

	"cuModuleLoad":        "Module Load",
	"cuModuleLoadData":    "Module LoadData",
	"cuModuleGetGlobal":   "Module Global",
	"cuModuleGetFunction": "Module Function",

	"cuModuleUnload": "Module Unload",

	"cuMemGetInfo":              "MemInfo",
	"cuMemAlloc":                "MemAlloc",
	"cuMemAllocPitch":           "MemAllocPitch",
	"cuMemFree":                 "MemFree",
	"cuMemGetAddressRange":      "DevicePtr AddressRange",
	"cuMemAllocHost":            "MemAllocHost",
	"cuMemFreeHost":             "MemFreeHost",
	"cuMemHostAlloc":            "MemHostAlloc",
	"cuMemHostGetDevicePointer": "MemHostGetDevicePointer",
	"cuMemHostGetFlags":         "MemHostGetFlags",
	"cuMemAllocManaged":         "MemAllocManaged",
	"cuMemcpy":                  "Memcpy",
	"cuMemcpyPeer":              "MemcpyPeer",
	"cuMemcpyHtoD":              "MemcpyHtoD",
	"cuMemcpyDtoH":              "MemcpyDtoH",
	"cuMemcpyDtoD":              "MemcpyDtoD",
	"cuMemcpyDtoA":              "MemcpyDtoA",
	"cuMemcpyAtoD":              "MemcpyAtoD",
	"cuMemcpyHtoA":              "MemcpyHtoA",
	"cuMemcpyAtoH":              "MemcpyAtoH",
	"cuMemcpyAtoA":              "MemcpyAtoA",
	"cuMemcpy2D":                "Memcpy2D",
	"cuMemcpy2DUnaligned":       "Memcpy2DUnaligned",
	"cuMemcpy3D":                "Memcpy3D",
	"cuMemcpy3DPeer":            "Memcpy3DPeer",
	"cuMemcpyAsync":             "MemcpyAsync",
	"cuMemcpyPeerAsync":         "MemcpyPeerAsync",
	"cuMemcpyHtoDAsync":         "MemcpyHtoDAsync",
	"cuMemcpyDtoHAsync":         "MemcpyDtoHAsync",
	"cuMemcpyDtoDAsync":         "MemcpyDtoDAsync",
	"cuMemcpyHtoAAsync":         "MemcpyHtoAAsync",
	"cuMemcpyAtoHAsync":         "MemcpyAtoHAsync",
	"cuMemcpy2DAsync":           "Memcpy2DAsync",
	"cuMemcpy3DAsync":           "Memcpy3DAsync",
	"cuMemcpy3DPeerAsync":       "Memcpy3DPeerAsync",
	"cuMemsetD8":                "MemsetD8",
	"cuMemsetD16":               "MemsetD16",
	"cuMemsetD32":               "MemsetD32",
	"cuMemsetD2D8":              "MemsetD2D8",
	"cuMemsetD2D16":             "MemsetD2D16",
	"cuMemsetD2D32":             "MemsetD2D32",
	"cuMemsetD8Async":           "MemsetD8Async",
	"cuMemsetD16Async":          "MemsetD16Async",
	"cuMemsetD32Async":          "MemsetD32Async",
	"cuMemsetD2D8Async":         "MemsetD2D8Async",
	"cuMemsetD2D16Async":        "MemsetD2D16Async",
	"cuMemsetD2D32Async":        "MemsetD2D32Async",

	"cuArrayCreate":          "MakeArray",
	"cuArrayGetDescriptor":   "Array Descriptor",
	"cuArrayDestroy":         "Array Destroy",
	"cuArray3DCreate":        "Make3DArray",
	"cuArray3DGetDescriptor": "Array Descriptor3",

	"cuStreamCreate":                    "MakeStream",
	"cuStreamCreateWithPriority":        "MakeStreamWithPriority",
	"cuStreamGetPriority":               "Stream Priority",
	"cuStreamGetFlags":                  "Stream Flags",
	"cuStreamWaitEvent":                 "Stream Wait",
	"cuStreamAddCallback":               "Stream AddCallback",
	"cuStreamAttachMemAsync":            "Stream AttachMemAsync",
	"cuStreamQuery":                     "Stream Query",
	"cuStreamSynchronize":               "Stream Synchronize",
	"cuStreamDestroy":                   "Stream Destroy",
	"cuStreamBeginCapture":              "Stream BeginCapture",
	"cuStreamCopyAttributes":            "Stream CopyAttributes",
	"cuStreamEndCapture":                "Stream EndCapture",
	"cuStreamGetAttribute":              "Stream Attribute",
	"cuStreamGetCaptureInfo":            "Stream CaptureInfo",
	"cuStreamGetCtx":                    "Stream Context",
	"cuStreamIsCapturing":               "Stream IsCapturing",
	"cuStreamSetAttribute":              "Stream SetAttribute",
	"cuStreamWaitValue64":               "Stream WaitOnValue64",
	"cuStreamWriteValue64":              "Stream WriteValue64",
	"cuThreadExchangeStreamCaptureMode": "ExchangeStreamCaptureThreads", // TODO - possibly manual write

	"cuEventCreate":        "MakeEvent",
	"cuEventRecord":        "Event Record",
	"cuEventQuery":         "Event Query",
	"cuEventSynchronize":   "Event Synchronize",
	"cuEventDestroy":       "Event Destroy",
	"cuEventElapsedTime":   "Event Elapsed", // getter
	"cuStreamWaitValue32":  "Stream WaitOnValue32",
	"cuStreamWriteValue32": "Stream WriteValue32",
	"cuStreamBatchMemOp":   "Stream BatchMemOp",

	"cuFuncGetAttribute":       "Function Attribute",
	"cuFuncSetAttribute":       "Function SetAttribute",
	"cuFuncSetCacheConfig":     "Function SetCacheConfig",
	"cuFuncSetSharedMemConfig": "Function SetSharedMemConfig",

	"cuOccupancyMaxActiveBlocksPerMultiprocessor":          "Function MaxActiveBlocksPerMultiProcessor",
	"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "Function MaxActiveBlocksPerMultiProcessorWithFlags",

	"cuTexRefSetArray":               "TexRef SetArray",
	"cuTexRefSetAddress":             "TexRef SetAddress",
	"cuTexRefSetAddress2D":           "TexRef SetAddress2D",
	"cuTexRefSetFormat":              "TexRef SetFormat",
	"cuTexRefSetAddressMode":         "TexRef SetAddressMode",
	"cuTexRefSetFilterMode":          "TexRef SetFilterMode",
	"cuTexRefSetMipmapFilterMode":    "TexRef SetMipmapFilterMode",
	"cuTexRefSetMipmapLevelBias":     "TexRef SetMipmapLevelBias",
	"cuTexRefSetMipmapLevelClamp":    "TexRef SetMipmapLevelClamp",
	"cuTexRefSetMaxAnisotropy":       "TexRef SetMaxAnisotropy",
	"cuTexRefSetBorderColor":         "TexRef SetBorderColor",
	"cuTexRefSetFlags":               "TexRef SetFlags",
	"cuTexRefGetAddress":             "TexRef Address",
	"cuTexRefGetArray":               "TexRef Array",
	"cuTexRefGetMipmappedArray":      "TexRef MipmappedArray",
	"cuTexRefGetAddressMode":         "TexRef AddressMode",
	"cuTexRefGetFilterMode":          "TexRef FilterMode",
	"cuTexRefGetFormat":              "TexRef Format",
	"cuTexRefGetMaxAnisotropy":       "TexRef MaxAnisotropy",
	"cuTexRefGetBorderColor":         "TexRef BorderColor",
	"cuTexRefGetFlags":               "TexRef Flags",
	"cuTexRefCreate":                 "MakeTexRef",
	"cuTexRefDestroy":                "DestroyTexRef",
	"cuSurfRefSetArray":              "SurfRef SetArray",
	"cuSurfRefGetArray":              "SurfRef GetArray",
	"cuTexObjectCreate":              "MakeTexObject",
	"cuTexObjectDestroy":             "DestroyTexObject",
	"cuTexObjectGetResourceDesc":     "TexObject GetResourceDesc",
	"cuTexObjectGetTextureDesc":      "TexObject GetTextureDesc",
	"cuTexObjectGetResourceViewDesc": "TexObject GetResourceViewDesc",
	"cuSurfObjectCreate":             "MakeSurfObject",
	"cuSurfObjectDestroy":            "DestroySurfObject",
	"cuSurfObjectGetResourceDesc":    "SurfObject GetResourceDesc",

	"cuDeviceCanAccessPeer":   "Device CanAccessPeer",
	"cuDeviceGetP2PAttribute": "Device P2PAttribute",
	"cuCtxEnablePeerAccess":   "CUContext EnablePeerAccess",
	"cuCtxDisablePeerAccess":  "CUContext DisablePeerAccess",
}

// list of functions that returns stuff but do not have "Get" in the name
var returns = []string{
	"cuDeviceCanAccessPeer",
	"cuDeviceGetP2PAttribute",
	"cuEventElapsedTime",
	"cuDeviceTotalMem",
	"cuCtxPopCurrent",

	"cuMemAlloc",
	"cuMemAllocPitch",
	"cuMemAllocHost",
	"cuMemAllocManaged",

	"cuTexRefSetAddress",
}

var ctypesFix = map[string]string{
	"unsigned int":   "uint",
	"unsigned char":  "uchar",
	"unsigned short": "ushort",
}

var ctypes2GoTypes = map[string]string{
	"C.CUdevice":                "Device",
	"C.CUdeviceptr":             "DevicePtr",
	"C.CUcontext":               "CUContext",
	"C.CUmodule":                "Module",
	"C.CUlimit":                 "Limit",
	"C.CUarray":                 "Array",
	"C.CUstream":                "Stream",
	"C.CUstreamCallback":        " StreamCallback",
	"C.CUevent":                 "Event",
	"C.CUfunction":              "Function",
	"C.CUtexref":                "TexRef",
	"C.CUarray_format":          "Format",
	"C.CUdevice_attribute":      "DeviceAttribute",
	"C.CUfunc_cache":            "FuncCacheConfig",
	"C.CUsurfref":               "SurfRef",
	"C.CUDA_MEMCPY2D":           "Memcpy2dParam",
	"C.CUDA_MEMCPY3D":           "Memcpy3dParam",
	"C.CUDA_MEMCPY3D_PEER":      "Memcpy3dPeerParam",
	"C.CUsharedconfig":          "SharedConfig",
	"C.CUDA_ARRAY_DESCRIPTOR":   "ArrayDesc",
	"C.CUDA_ARRAY3D_DESCRIPTOR": "Array3Desc",
	"C.CUfunction_attribute":    "FunctionAttribute",
	"C.CUaddress_mode":          "AddressMode",
	"C.CUfilter_mode":           "FilterMode",
	"C.CUdevice_P2PAttribute":   "P2PAttribute",

	"C.CUgraph":         "Graph",
	"C.CUgraphExec":     "ExecGraph",
	"C.CUgraphNodeType": "Node",

	"C.cuuint32_t": "uint32",
	"C.cuuint64_t": "uint64",

	"C.uint":   "uint",
	"C.uchar":  "byte",
	"C.char":   "byte",
	"C.ushort": "uint16",
	"C.size_t": "int64",
	"C.int":    "int",
	"C.float":  "float64",
	"C.void":   "unsafe.Pointer",
	"C.void*":  "*unsafe.Pointer",

	"C.unsigned":           "uint",
	"C.unsigned char":      "byte",
	"C.unsigned short":     "uint16",
	"C.unsigned long long": "uint64",
}

var gotypesConversion = map[string]string{
	"Device":            "C.CUdevice(%s)",
	"DevicePtr":         "C.CUdeviceptr(%s)",
	"CUContext":         "%s.c()",
	"Module":            "%s.c()",
	"Array":             "%s.c()",
	"Stream":            "%s.c()",
	"Event":             "%s.c()",
	"Function":          "%s.c()",
	"TexRef":            "%s.c()",
	"SurfRef":           "%s.c()",
	"DeviceAttribute":   "C.CUdevice_attribute(%s)",
	"P2PAttribute":      "C.CUdevice_P2PAttribute(%s)",
	"FunctionAttribute": "C.CUfunction_attribute(%s)",
	"Memcpy2dParam":     "%s.c()",
	"Memcpy3dParam":     "%s.c()",
	"Memcpy3dPeerParam": "%s.c()",
	"ArrayDesc":         "%s.c()",
	"Array3Desc":        "%s.c()",

	"Graph":     "%s.c()",
	"ExecGraph": "%s.c()",
	"Node":      "%s.c()",

	// flags, which are mostly uint in the C signature
	"Format":          "C.CUarray_format(%s)",
	"FuncCacheConfig": "C.CUfunc_cache(%s)",
	"Limit":           "C.CUlimit(%s)",
	"MemAttachFlag":   "C.uint(%s)",
	"StreamFlags":     "C.uint(%s)",
	"AddressMode":     "C.CUaddress_mode(%s)",
	"FilterMode":      "C.CUfilter_mode(%s)",
	"ContextFlags":    "C.CUctx_flags(%s)",
	"SharedConfig":    "C.CUsharedconfig(%s)",
	"EventFlags":      "C.CUevent_flags(%s)",
	"TexRefFlags":     "C.uint(%s)",

	"uint":            "C.uint(%s)",
	"byte":            "C.uchar(%s)",
	"uint16":          "C.ushort(%s)",
	"uint32":          "C.cuuint32_t(%s)", // there is only one uint32
	"uint64":          "C.cuuint64_t(%s)", // there are two uint64s, but both works because C.
	"int":             "C.int(%s)",
	"int64":           "C.size_t(%s)",
	"float64":         "C.float(%s)", // there is only one instance of float64
	"unsafe.Pointer":  "%s",
	"*unsafe.Pointer": "%s",
}

var ctypesConversion = map[string]string{
	//"C.CUstream":                "Stream(uintptr(unsafe.Pointer(%s)))",
	//"C.CUevent":                 "Event(uintptr(unsafe.Pointer(%s)))",
	"C.CUDA_ARRAY_DESCRIPTOR":   "goArrayDesc(&%s)",
	"C.CUDA_ARRAY3D_DESCRIPTOR": "goArray3Desc(&%s)",
	"C.CUarray":                 "goArray(&%s)",
	"C.CUcontext":               "makeContext(%s)",
}

var renames = map[string]string{
	"func":  "fn",
	"hFunc": "fn",
	"hfunc": "fn",
}
