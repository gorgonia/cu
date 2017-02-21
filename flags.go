package cu

//#include <cuda.h>
import "C"

// FuncCacheConfig represents the CUfunc_cache enum type, which are enumerations for cache configurations
type FuncCacheConfig byte

const (
	PreferNone   FuncCacheConfig = C.CU_FUNC_CACHE_PREFER_NONE   // no preference for shared memory or L1 (default)
	PreferShared FuncCacheConfig = C.CU_FUNC_CACHE_PREFER_SHARED // prefer larger shared memory and smaller L1 cache
	PreferL1     FuncCacheConfig = C.CU_FUNC_CACHE_PREFER_L1     // prefer larger L1 cache and smaller shared memory
	PreferEqual  FuncCacheConfig = C.CU_FUNC_CACHE_PREFER_EQUAL  // prefer equal sized L1 cache and shared memory
)

// ContextFlags are flags that are used to create a context
type ContextFlags byte

const (
	SchedAuto         ContextFlags = C.CU_CTX_SCHED_AUTO          // Automatic scheduling
	SchedSpin         ContextFlags = C.CU_CTX_SCHED_SPIN          // Set spin as default scheduling
	SchedYield        ContextFlags = C.CU_CTX_SCHED_YIELD         // Set yield as default scheduling
	SchedBlockingSync ContextFlags = C.CU_CTX_SCHED_BLOCKING_SYNC // Set blocking synchronization as default scheduling
	SchedMask         ContextFlags = C.CU_CTX_SCHED_MASK          // Mask for setting scheduling options for the flag
	MapHost           ContextFlags = C.CU_CTX_MAP_HOST            // Support mapped pinned allocations
	LMemResizeToMax   ContextFlags = C.CU_CTX_LMEM_RESIZE_TO_MAX  // Keep local memory allocation after launch
	FlagsMas          ContextFlags = C.CU_CTX_FLAGS_MASK          // Mask for setting other options to flags
)

// Limit is a flag that can be used to query and set on a context
type Limit byte

const (
	StackSize                    Limit = C.CU_LIMIT_STACK_SIZE                       // GPU thread stack size
	PrintfFIFOSize               Limit = C.CU_LIMIT_PRINTF_FIFO_SIZE                 // GPU printf FIFO size
	MallocHeapSize               Limit = C.CU_LIMIT_MALLOC_HEAP_SIZE                 // GPU malloc heap size
	DevRuntimeSyncDepth          Limit = C.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           // GPU device runtime launch synchronize depth
	DevRuntimePendingLaunchCount Limit = C.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT // GPU device runtime pending launch count
)

// ShareConfigs are flags for shared memory configurations
type SharedConfig byte

const (
	DefaultBankSize   SharedConfig = C.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    // set default shared memory bank size
	FourByteBankSize  SharedConfig = C.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  // set shared memory bank width to four bytes
	EightByteBankSize SharedConfig = C.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE // set shared memory bank width to eight bytes
)

// MemAttachFlags are flags for memory attachment (used in allocating memory)
type MemAttachFlags byte

const (
	AttachGlobal MemAttachFlags = C.CU_MEM_ATTACH_GLOBAL // Memory can be accessed by any stream on any device
	AttachHost   MemAttachFlags = C.CU_MEM_ATTACH_HOST   // Memory cannot be accessed by any stream on any device
	AttachSingle MemAttachFlags = C.CU_MEM_ATTACH_SINGLE // Memory can only be accessed by a single stream on the associated device
)

// StreamFlags are flags for stream behaviours
type StreamFlags byte

const (
	DefaultStream StreamFlags = C.CU_STREAM_DEFAULT      // Default stream flag
	NonBlocking   StreamFlags = C.CU_STREAM_NON_BLOCKING // Stream does not synchronize with stream 0 (the NULL stream)
)

// MemAdvice is a flag that advises the device on memory usage
type MemAdvice byte

const (
	SetReadMostly          MemAdvice = C.CU_MEM_ADVISE_SET_READ_MOSTLY          // Data will mostly be read and only occassionally be written to
	UnsetReadMostly        MemAdvice = C.CU_MEM_ADVISE_UNSET_READ_MOSTLY        // Undo the effect of CU_MEM_ADVISE_SET_READ_MOSTLY
	SetPreferredLocation   MemAdvice = C.CU_MEM_ADVISE_SET_PREFERRED_LOCATION   // Set the preferred location for the data as the specified device
	UnsetPreferredLocation MemAdvice = C.CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION // Clear the preferred location for the data
	SetAccessedBy          MemAdvice = C.CU_MEM_ADVISE_SET_ACCESSED_BY          // Data will be accessed by the specified device, so prevent page faults as much as possible
	UnsetAccessedBy        MemAdvice = C.CU_MEM_ADVISE_UNSET_ACCESSED_BY        //Let the Unified Memory subsystem decide on the page faulting policy for the specified device
)

// MemoryType is a representation of the memory types of the device pointer
type MemoryType byte

const (
	HostMemory    MemoryType = C.CU_MEMORYTYPE_HOST    // Host memory
	DeviceMemory  MemoryType = C.CU_MEMORYTYPE_DEVICE  // Device memory
	ArrayMemory   MemoryType = C.CU_MEMORYTYPE_ARRAY   // Array memory
	UnifiedMemory MemoryType = C.CU_MEMORYTYPE_UNIFIED // Unified device or host memory
)

// OccupanyFlags represents the flags to the occupancy calculator
type OccupancyFlags byte

const (
	DefaultOccupancy       OccupancyFlags = C.CU_OCCUPANCY_DEFAULT                  // Default behavior
	DisableCachingOverride OccupancyFlags = C.CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE // Assume global caching is enabled and cannot be automatically turned off
)
