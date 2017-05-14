package cu

//#include <cuda.h>
import "C"

// DeviceAttribute represents the device attributes that the user can query CUDA for.
type DeviceAttribute int

const (
	MaxThreadsPerBlock                 DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                   // Maximum number of threads per block
	MaxBlockDimX                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                         // Maximum block dimension X
	MaxBlockDimY                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                         // Maximum block dimension Y
	MaxBlockDimZ                       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                         // Maximum block dimension Z
	MaxGridDimX                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                          // Maximum grid dimension X
	MaxGridDimY                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                          // Maximum grid dimension Y
	MaxGridDimZ                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                          // Maximum grid dimension Z
	MaxSharedMemoryPerBlock            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK             // Maximum shared memory available per block in bytes
	SharedMemoryPerBlock               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                 // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
	TotalConstantMemory                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                   // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
	WarpSize                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_WARP_SIZE                               // Warp size in threads
	MaxPitch                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_PITCH                               // Maximum pitch in bytes allowed by memory copies
	MaxRegistersPerBlock               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                 // Maximum number of 32-bit registers available per block
	RegistersPerBlock                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                     // Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
	ClockRate                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CLOCK_RATE                              // Typical clock frequency in kilohertz
	TextureAlignment                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                       // Alignment requirement for textures
	GpuOverlap                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                             // Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
	MultiprocessorCount                DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                    // Number of multiprocessors on device
	KernelExecTimeout                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                     // Specifies whether there is a run time limit on kernels
	Integrated                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_INTEGRATED                              // Device is integrated with host memory
	CanMapHostMemory                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                     // Device can map host memory into CUDA address space
	ComputeMode                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                            // Compute mode (See CUcomputemode for details)
	MaximumTexture1dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                 // Maximum 1D texture width
	MaximumTexture2dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                 // Maximum 2D texture width
	MaximumTexture2dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                // Maximum 2D texture height
	MaximumTexture3dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                 // Maximum 3D texture width
	MaximumTexture3dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                // Maximum 3D texture height
	MaximumTexture3dDepth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                 // Maximum 3D texture depth
	MaximumTexture2dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH         // Maximum 2D layered texture width
	MaximumTexture2dLayeredHeight      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT        // Maximum 2D layered texture height
	MaximumTexture2dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS        // Maximum layers in a 2D layered texture
	MaximumTexture2dArrayWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH           // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
	MaximumTexture2dArrayHeight        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT          // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
	MaximumTexture2dArrayNumslices     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES       // Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
	SurfaceAlignment                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                       // Alignment requirement for surfaces
	ConcurrentKernels                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                      // Device can possibly execute multiple kernels concurrently
	EccEnabled                         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ECC_ENABLED                             // Device has ECC support enabled
	PciBusID                           DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                              // PCI bus ID of the device
	PciDeviceID                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                           // PCI device ID of the device
	TccDriver                          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TCC_DRIVER                              // Device is using TCC driver model
	MemoryClockRate                    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                       // Peak memory clock frequency in kilohertz
	GlobalMemoryBusWidth               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                 // Global memory bus width in bits
	L2CacheSize                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                           // Size of L2 cache in bytes
	MaxThreadsPerMultiprocessor        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR          // Maximum resident threads per multiprocessor
	AsyncEngineCount                   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                      // Number of asynchronous engines
	UnifiedAddressing                  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                      // Device shares a unified address space with the host
	MaximumTexture1dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH         // Maximum 1D layered texture width
	MaximumTexture1dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS        // Maximum layers in a 1D layered texture
	CanTex2dGather                     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                        // Deprecated, do not use.
	MaximumTexture2dGatherWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH          // Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture2dGatherHeight       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT         // Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
	MaximumTexture3dWidthAlternate     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE       // Alternate maximum 3D texture width
	MaximumTexture3dHeightAlternate    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE      // Alternate maximum 3D texture height
	MaximumTexture3dDepthAlternate     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE       // Alternate maximum 3D texture depth
	PciDomainID                        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                           // PCI domain ID of the device
	TexturePitchAlignment              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                 // Pitch alignment requirement for textures
	MaximumTexturecubemapWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH            // Maximum cubemap texture width/height
	MaximumTexturecubemapLayeredWidth  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH    // Maximum cubemap layered texture width/height
	MaximumTexturecubemapLayeredLayers DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS   // Maximum layers in a cubemap layered texture
	MaximumSurface1dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                 // Maximum 1D surface width
	MaximumSurface2dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                 // Maximum 2D surface width
	MaximumSurface2dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                // Maximum 2D surface height
	MaximumSurface3dWidth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                 // Maximum 3D surface width
	MaximumSurface3dHeight             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                // Maximum 3D surface height
	MaximumSurface3dDepth              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                 // Maximum 3D surface depth
	MaximumSurface1dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH         // Maximum 1D layered surface width
	MaximumSurface1dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS        // Maximum layers in a 1D layered surface
	MaximumSurface2dLayeredWidth       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH         // Maximum 2D layered surface width
	MaximumSurface2dLayeredHeight      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT        // Maximum 2D layered surface height
	MaximumSurface2dLayeredLayers      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS        // Maximum layers in a 2D layered surface
	MaximumSurfacecubemapWidth         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH            // Maximum cubemap surface width
	MaximumSurfacecubemapLayeredWidth  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH    // Maximum cubemap layered surface width
	MaximumSurfacecubemapLayeredLayers DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS   // Maximum layers in a cubemap layered surface
	MaximumTexture1dLinearWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH          // Maximum 1D linear texture width
	MaximumTexture2dLinearWidth        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH          // Maximum 2D linear texture width
	MaximumTexture2dLinearHeight       DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT         // Maximum 2D linear texture height
	MaximumTexture2dLinearPitch        DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH          // Maximum 2D linear texture pitch in bytes
	MaximumTexture2dMipmappedWidth     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH       // Maximum mipmapped 2D texture width
	MaximumTexture2dMipmappedHeight    DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT      // Maximum mipmapped 2D texture height
	ComputeCapabilityMajor             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                // Major compute capability version number
	ComputeCapabilityMinor             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                // Minor compute capability version number
	MaximumTexture1dMipmappedWidth     DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH       // Maximum mipmapped 1D texture width
	StreamPrioritiesSupported          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED             // Device supports stream priorities
	GlobalL1CacheSupported             DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED               // Device supports caching globals in L1
	LocalL1CacheSupported              DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                // Device supports caching locals in L1
	MaxSharedMemoryPerMultiprocessor   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR    // Maximum shared memory available per multiprocessor in bytes
	MaxRegistersPerMultiprocessor      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR        // Maximum number of 32-bit registers available per multiprocessor
	ManagedMemory                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                          // Device can allocate managed memory on this system
	MultiGpuBoard                      DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                         // Device is on a multi-GPU board
	MultiGpuBoardGroupID               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                // Unique id for a group of devices on the same multi-GPU board
	HostNativeAtomicSupported          DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED            // Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
	SingleToDoublePrecisionPerfRatio   DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO   // Ratio of single precision performance (in floating-point operations per second) to double precision performance
	PageableMemoryAccess               DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                  // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
	ConcurrentManagedAccess            DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS               // Device can coherently access managed memory concurrently with the CPU
	ComputePreemptionSupported         DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED            // Device supports compute preemption.
	CanUseHostPointerForRegisteredMem  DeviceAttribute = C.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM // Device can access host registered memory at the same virtual address as the CPU

)

// FunctionAttribute is a representation of the properties of a function
type FunctionAttribute int

const (
	FnMaxThreadsPerBlock FunctionAttribute = C.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK // The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.
	SharedSizeBytes      FunctionAttribute = C.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     // The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.
	ConstSizeBytes       FunctionAttribute = C.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      // The size in bytes of user-allocated constant memory required by this function.
	LocalSizeBytes       FunctionAttribute = C.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      // The size in bytes of local memory used by each thread of this function.
	NumRegs              FunctionAttribute = C.CU_FUNC_ATTRIBUTE_NUM_REGS              // The number of registers used by each thread of this function.
	PtxVersion           FunctionAttribute = C.CU_FUNC_ATTRIBUTE_PTX_VERSION           // The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.
	BinaryVersion        FunctionAttribute = C.CU_FUNC_ATTRIBUTE_BINARY_VERSION        // The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.
	CacheModeCa          FunctionAttribute = C.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA         // The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set .
)

// PointerAttribute is a representation of the metadata of pointers
type PointerAttribute int

const (
	ContextAttr       PointerAttribute = C.CU_POINTER_ATTRIBUTE_CONTEXT        // The CUcontext on which a pointer was allocated or registered
	MemoryTypeAttr    PointerAttribute = C.CU_POINTER_ATTRIBUTE_MEMORY_TYPE    // The CUmemorytype describing the physical location of a pointer
	DevicePointerAttr PointerAttribute = C.CU_POINTER_ATTRIBUTE_DEVICE_POINTER // The address at which a pointer's memory may be accessed on the device
	HostPointerAttr   PointerAttribute = C.CU_POINTER_ATTRIBUTE_HOST_POINTER   // The address at which a pointer's memory may be accessed on the host
	P2PTokenAttr      PointerAttribute = C.CU_POINTER_ATTRIBUTE_P2P_TOKENS     // A pair of tokens for use with the nv-p2p.h Linux kernel interface
	SymcMemopsAttr    PointerAttribute = C.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS    // Synchronize every synchronous memory operation initiated on this region
	BufferIDAttr      PointerAttribute = C.CU_POINTER_ATTRIBUTE_BUFFER_ID      // A process-wide unique ID for an allocated memory region
	IsManagedAttr     PointerAttribute = C.CU_POINTER_ATTRIBUTE_IS_MANAGED     // Indicates if the pointer points to managed memory
)

// P2PAttribute is a representation of P2P attributes
type P2PAttribute byte

const (
	PerformanceRank         P2PAttribute = C.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK        // A relative value indicating the performance of the link between two devices
	P2PAccessSupported      P2PAttribute = C.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED        // P2P Access is enabled
	P2PNativeAomicSupported P2PAttribute = C.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED // Atomic operation over the link is supported
)
