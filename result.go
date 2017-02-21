package cu

//#include <cuda.h>
import "C"

// cuResult is the Go version of CUresult:
// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9
type cuResult int

func (err cuResult) Error() string  { return err.String() }
func (err cuResult) String() string { return resString[err] }

func result(x C.CUresult) error {
	err := cuResult(x)
	if err == Success {
		return nil
	}
	if err > Unknown {
		return Unknown
	}

	return err
}

const (
	Success                     cuResult = C.CUDA_SUCCESS                              // API call returned with no errors
	InvalidValue                cuResult = C.CUDA_ERROR_INVALID_VALUE                  // This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
	OutOfMemory                 cuResult = C.CUDA_ERROR_OUT_OF_MEMORY                  // The API call failed because it was unable to allocate enough memory to perform the requested operation.
	NotInitialized              cuResult = C.CUDA_ERROR_NOT_INITIALIZED                // This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed.
	Deinitialized               cuResult = C.CUDA_ERROR_DEINITIALIZED                  // This indicates that the CUDA driver is in the process of shutting down.
	ProfilerDisabled            cuResult = C.CUDA_ERROR_PROFILER_DISABLED              // This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
	ProfilerNotInitialized      cuResult = C.CUDA_ERROR_PROFILER_NOT_INITIALIZED       // Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via cuProfilerStart or cuProfilerStop without initialization.
	ProfilerAlreadyStarted      cuResult = C.CUDA_ERROR_PROFILER_ALREADY_STARTED       // Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() when profiling is already enabled.
	ProfilerAlreadyStopped      cuResult = C.CUDA_ERROR_PROFILER_ALREADY_STOPPED       // Deprecated: This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() when profiling is already disabled.
	NoDevice                    cuResult = C.CUDA_ERROR_NO_DEVICE                      // This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
	InvalidDevice               cuResult = C.CUDA_ERROR_INVALID_DEVICE                 // This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.
	InvalidImage                cuResult = C.CUDA_ERROR_INVALID_IMAGE                  // This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.
	InvalidContext              cuResult = C.CUDA_ERROR_INVALID_CONTEXT                // This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
	ContextAlreadyCurrent       cuResult = C.CUDA_ERROR_CONTEXT_ALREADY_CURRENT        // Deprecated: This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active context via cuCtxPushCurrent(). This indicated that the context being supplied as a parameter to the API call was already the active context.
	MapFailed                   cuResult = C.CUDA_ERROR_MAP_FAILED                     // This indicates that a map or register operation has failed.
	UnmapFailed                 cuResult = C.CUDA_ERROR_UNMAP_FAILED                   // This indicates that an unmap or unregister operation has failed.
	ArrayIsMapped               cuResult = C.CUDA_ERROR_ARRAY_IS_MAPPED                // This indicates that the specified array is currently mapped and thus cannot be destroyed.
	AlreadyMapped               cuResult = C.CUDA_ERROR_ALREADY_MAPPED                 // This indicates that the resource is already mapped.
	NoBinaryForGpu              cuResult = C.CUDA_ERROR_NO_BINARY_FOR_GPU              // This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
	AlreadyAcquired             cuResult = C.CUDA_ERROR_ALREADY_ACQUIRED               // This indicates that a resource has already been acquired.
	NotMapped                   cuResult = C.CUDA_ERROR_NOT_MAPPED                     // This indicates that a resource is not mapped.
	NotMappedAsArray            cuResult = C.CUDA_ERROR_NOT_MAPPED_AS_ARRAY            // This indicates that a mapped resource is not available for access as an array.
	NotMappedAsPointer          cuResult = C.CUDA_ERROR_NOT_MAPPED_AS_POINTER          // This indicates that a mapped resource is not available for access as a pointer.
	EccUncorrectable            cuResult = C.CUDA_ERROR_ECC_UNCORRECTABLE              // This indicates that an uncorrectable ECC error was detected during execution.
	UnsupportedLimit            cuResult = C.CUDA_ERROR_UNSUPPORTED_LIMIT              // This indicates that the CUlimit passed to the API call is not supported by the active device.
	ContextAlreadyInUse         cuResult = C.CUDA_ERROR_CONTEXT_ALREADY_IN_USE         // This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.
	PeerAccessUnsupported       cuResult = C.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        // This indicates that peer access is not supported across the given devices.
	InvalidPtx                  cuResult = C.CUDA_ERROR_INVALID_PTX                    // This indicates that a PTX JIT compilation failed.
	InvalidGraphicsContext      cuResult = C.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       // This indicates an error with OpenGL or DirectX context.
	NvlinkUncorrectable         cuResult = C.CUDA_ERROR_NVLINK_UNCORRECTABLE           // This indicates that an uncorrectable NVLink error was detected during the execution.
	InvalidSource               cuResult = C.CUDA_ERROR_INVALID_SOURCE                 // This indicates that the device kernel source is invalid.
	FileNotFound                cuResult = C.CUDA_ERROR_FILE_NOT_FOUND                 // This indicates that the file specified was not found.
	SharedObjectSymbolNotFound  cuResult = C.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND // This indicates that a link to a shared object failed to resolve.
	SharedObjectInitFailed      cuResult = C.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      // This indicates that initialization of a shared object failed.
	OperatingSystem             cuResult = C.CUDA_ERROR_OPERATING_SYSTEM               // This indicates that an OS call failed.
	InvalidHandle               cuResult = C.CUDA_ERROR_INVALID_HANDLE                 // This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent.
	NotFound                    cuResult = C.CUDA_ERROR_NOT_FOUND                      // This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names.
	NotReady                    cuResult = C.CUDA_ERROR_NOT_READY                      // This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
	IllegalAddress              cuResult = C.CUDA_ERROR_ILLEGAL_ADDRESS                // While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	LaunchOutOfResources        cuResult = C.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        // This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.
	LaunchTimeout               cuResult = C.CUDA_ERROR_LAUNCH_TIMEOUT                 // This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	LaunchIncompatibleTexturing cuResult = C.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  // This error indicates a kernel launch that uses an incompatible texturing mode.
	PeerAccessAlreadyEnabled    cuResult = C.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    // This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled.
	PeerAccessNotEnabled        cuResult = C.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        // This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess().
	PrimaryContextActive        cuResult = C.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         // This error indicates that the primary context for the specified device has already been initialized.
	ContextIsDestroyed          cuResult = C.CUDA_ERROR_CONTEXT_IS_DESTROYED           // This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized.
	Assert                      cuResult = C.CUDA_ERROR_ASSERT                         // A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.
	TooManyPeers                cuResult = C.CUDA_ERROR_TOO_MANY_PEERS                 // This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess().
	HostMemoryAlreadyRegistered cuResult = C.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED // This error indicates that the memory range passed to cuMemHostRegister() has already been registered.
	HostMemoryNotRegistered     cuResult = C.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     // This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region.
	HardwareStackError          cuResult = C.CUDA_ERROR_HARDWARE_STACK_ERROR           // While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	IllegalInstruction          cuResult = C.CUDA_ERROR_ILLEGAL_INSTRUCTION            // While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	MisalignedAddress           cuResult = C.CUDA_ERROR_MISALIGNED_ADDRESS             // While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	InvalidAddressSpace         cuResult = C.CUDA_ERROR_INVALID_ADDRESS_SPACE          // While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	InvalidPc                   cuResult = C.CUDA_ERROR_INVALID_PC                     // While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	LaunchFailed                cuResult = C.CUDA_ERROR_LAUNCH_FAILED                  // An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
	NotPermitted                cuResult = C.CUDA_ERROR_NOT_PERMITTED                  // This error indicates that the attempted operation is not permitted.
	NotSupported                cuResult = C.CUDA_ERROR_NOT_SUPPORTED                  // This error indicates that the attempted operation is not supported on the current system or device.
	Unknown                     cuResult = C.CUDA_ERROR_UNKNOWN                        // This indicates that an unknown internal error has occurred.
)

var resString = map[cuResult]string{
	Success:                     "Success",
	InvalidValue:                "InvalidValue",
	OutOfMemory:                 "OutOfMemory",
	NotInitialized:              "NotInitialized",
	Deinitialized:               "Deinitialized",
	ProfilerDisabled:            "ProfilerDisabled",
	ProfilerNotInitialized:      "ProfilerNotInitialized",
	ProfilerAlreadyStarted:      "ProfilerAlreadyStarted",
	ProfilerAlreadyStopped:      "ProfilerAlreadyStopped",
	NoDevice:                    "NoDevice",
	InvalidDevice:               "InvalidDevice",
	InvalidImage:                "InvalidImage",
	InvalidContext:              "InvalidContext",
	ContextAlreadyCurrent:       "ContextAlreadyCurrent",
	MapFailed:                   "MapFailed",
	UnmapFailed:                 "UnmapFailed",
	ArrayIsMapped:               "ArrayIsMapped",
	AlreadyMapped:               "AlreadyMapped",
	NoBinaryForGpu:              "NoBinaryForGpu",
	AlreadyAcquired:             "AlreadyAcquired",
	NotMapped:                   "NotMapped",
	NotMappedAsArray:            "NotMappedAsArray",
	NotMappedAsPointer:          "NotMappedAsPointer",
	EccUncorrectable:            "EccUncorrectable",
	UnsupportedLimit:            "UnsupportedLimit",
	ContextAlreadyInUse:         "ContextAlreadyInUse",
	PeerAccessUnsupported:       "PeerAccessUnsupported",
	InvalidPtx:                  "InvalidPtx",
	InvalidGraphicsContext:      "InvalidGraphicsContext",
	NvlinkUncorrectable:         "NvlinkUncorrectable",
	InvalidSource:               "InvalidSource",
	FileNotFound:                "FileNotFound",
	SharedObjectSymbolNotFound:  "SharedObjectSymbolNotFound",
	SharedObjectInitFailed:      "SharedObjectInitFailed",
	OperatingSystem:             "OperatingSystem",
	InvalidHandle:               "InvalidHandle",
	NotFound:                    "NotFound",
	NotReady:                    "NotReady",
	IllegalAddress:              "IllegalAddress",
	LaunchOutOfResources:        "LaunchOutOfResources",
	LaunchTimeout:               "LaunchTimeout",
	LaunchIncompatibleTexturing: "LaunchIncompatibleTexturing",
	PeerAccessAlreadyEnabled:    "PeerAccessAlreadyEnabled",
	PeerAccessNotEnabled:        "PeerAccessNotEnabled",
	PrimaryContextActive:        "PrimaryContextActive",
	ContextIsDestroyed:          "ContextIsDestroyed",
	Assert:                      "Assert",
	TooManyPeers:                "TooManyPeers",
	HostMemoryAlreadyRegistered: "HostMemoryAlreadyRegistered",
	HostMemoryNotRegistered:     "HostMemoryNotRegistered",
	HardwareStackError:          "HardwareStackError",
	IllegalInstruction:          "IllegalInstruction",
	MisalignedAddress:           "MisalignedAddress",
	InvalidAddressSpace:         "InvalidAddressSpace",
	InvalidPc:                   "InvalidPc",
	LaunchFailed:                "LaunchFailed",
	NotPermitted:                "NotPermitted",
	NotSupported:                "NotSupported",
	Unknown:                     "Unknown",
}
