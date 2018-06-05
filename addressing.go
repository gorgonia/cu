package cu

// #include <cuda.h>
import "C"
import "unsafe"

// READ THIS PAGE: http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html

// MemAdvise advises the Unified Memory subsystem about the usage pattern for the memory range starting at d with a size of count bytes.
// The start address and end address of the memory range will be rounded down and rounded up respectively to be aligned to CPU page size before the advice is applied.
// The memory range must refer to managed memory allocated via `MemAllocManaged` or declared via __managed__ variables.
//
// The advice parameters can take either of the following values:
//		- SetReadMostly:
//			This implies that the data is mostly going to be read from and only occasionally written to.
//			Any read accesses from any processor to this region will create a read-only copy of at least the accessed pages in that processor's memory. Additionally, if cuMemPrefetchAsync is called on this region, it will create a read-only copy of the data on the destination processor. If any processor writes to this region, all copies of the corresponding page will be invalidated except for the one where the write occurred. The device argument is ignored for this advice. Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU that has a non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Also, if a context is created on a device that does not have the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS set, then read-duplication will not occur until all such contexts are destroyed.
// 		- UnsetReadMostly:
//			Undoes the effect of SetReadMostly and also prevents the Unified Memory driver from attempting heuristic read-duplication on the memory range.
// 			Any read-duplicated copies of the data will be collapsed into a single copy.
//			The location for the collapsed copy will be the preferred location if the page has a preferred location and one of the read-duplicated copies was resident at that location.
//			Otherwise, the location chosen is arbitrary.
//		- SetPreferredLocation:
// 			This advice sets the preferred location for the data to be the memory belonging to device.
// 			Passing in CU_DEVICE_CPU for device sets the preferred location as host memory.
// 			If device is a GPU, then it must have a non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
// 			Setting the preferred location does not cause data to migrate to that location immediately.
//			Instead, it guides the migration policy when a fault occurs on that memory region.
//			If the data is already in its preferred location and the faulting processor can establish a mapping without requiring the data to be migrated, then data migration will be avoided.
// 			On the other hand, if the data is not in its preferred location or if a direct mapping cannot be established, then it will be migrated to the processor accessing it.
// 			It is important to note that setting the preferred location does not prevent data prefetching done using cuMemPrefetchAsync.
// 			Having a preferred location can override the page thrash detection and resolution logic in the Unified Memory driver.
// 			Normally, if a page is detected to be constantly thrashing between for example host and device memory, the page may eventually be pinned to host memory by the Unified Memory driver.
// 			But if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
// 			If CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the policies associated with that advice will override the policies of this advice.
//		- UnsetPreferredLocation:
// 			Undoes the effect of SetPreferredLocation and changes the preferred location to none.
//		- SetAccessedBy:
//			This advice implies that the data will be accessed by device.
// 			Passing in CU_DEVICE_CPU for device will set the advice for the CPU.
// 			If device is a GPU, then the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS must be non-zero.
// 			This advice does not cause data migration and has no impact on the location of the data per se.
// 			Instead, it causes the data to always be mapped in the specified processor's page tables, as long as the location of the data permits a mapping to be established.
// 			If the data gets migrated for any reason, the mappings are updated accordingly.
//			This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
// 			Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the data located on one GPU is occasionally accessed by peer GPUs.
// 			In such scenarios, migrating data over to the other GPUs is not as important because the accesses are infrequent and the overhead of migration may be too high.
// 			But preventing faults can still help improve performance, and so having a mapping set up in advance is useful.
// 			Note that on CPU access of this data, the data may be migrated to host memory because the CPU typically cannot access device memory directly.
// 			Any GPU that had the CU_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the page in host memory.
//			If CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the policies associated with that advice will override the policies of this advice.
// 			Additionally, if the preferred location of this memory region or any subset of it is also device, then the policies associated with CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
//		- UnsetAccessedBy:
// 			Undoes the effect of SetAccessedBy.
// 			Any mappings to the data from device may be removed at any time causing accesses to result in non-fatal page faults.
func (d DevicePtr) MemAdvise(count int64, advice MemAdvice, dev Device) error {
	devPtr := C.CUdeviceptr(d)
	ad := C.CUmem_advise(advice)
	dv := C.CUdevice(dev)
	cc := C.size_t(count)
	return result(C.cuMemAdvise(devPtr, cc, ad, dv))
}

// MemPrefetchAsync prefetches memory to the specified destination device. devPtr is the base device pointer of the memory to be prefetched and dstDevice is the destination device. count specifies the number of bytes to copy. hStream is the stream in which the operation is enqueued. The memory range must refer to managed memory allocated via cuMemAllocManaged or declared via __managed__ variables.
// Passing in CU_DEVICE_CPU for dstDevice will prefetch the data to host memory. If dstDevice is a GPU, then the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS must be non-zero. Additionally, hStream must be associated with a device that has a non-zero value for the device attribute CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
//
// The start address and end address of the memory range will be rounded down and rounded up respectively to be aligned to CPU page size before the prefetch operation is enqueued in the stream.
//
// If no physical memory has been allocated for this region, then this memory region will be populated and mapped on the destination device. If there's insufficient memory to prefetch the desired region, the Unified Memory driver may evict pages from other cuMemAllocManaged allocations to host memory in order to make room. Device memory allocated using cuMemAlloc or cuArrayCreate will not be evicted.
//
//By default, any mappings to the previous location of the migrated pages are removed and mappings for the new location are only setup on dstDevice. The exact behavior however also depends on the settings applied to this memory range via cuMemAdvise as described below:
//
// If CU_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range, then that subset will create a read-only copy of the pages on dstDevice.
//
// If CU_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory range, then the pages will be migrated to dstDevice even if dstDevice is not the preferred location of any pages in the memory range.
//
// If CU_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range, then mappings to those pages from all the appropriate processors are updated to refer to the new location if establishing such a mapping is possible. Otherwise, those mappings are cleared.
//
// Note that this API is not required for functionality and only serves to improve performance by allowing the application to migrate data to a suitable location before it is accessed. Memory accesses to this range are always coherent and are allowed even when the data is actively being migrated.
//
// Note that this function is asynchronous with respect to the host and all work on other devices.
func (d DevicePtr) MemPrefetchAsync(count int64, dst Device, hStream Stream) error {
	devPtr := C.CUdeviceptr(d)
	cc := C.size_t(count)
	str := hStream.s
	dv := C.CUdevice(dst)
	return result(C.cuMemPrefetchAsync(devPtr, cc, dv, str))
}

// PtrAttribute returns information about a pointer.
func (d DevicePtr) PtrAttribute(attr PointerAttribute) (unsafe.Pointer, error) {
	var p unsafe.Pointer
	devPtr := C.CUdeviceptr(d)
	a := C.CUpointer_attribute(attr)
	if err := result(C.cuPointerGetAttribute(p, a, devPtr)); err != nil {
		return nil, err
	}
	return p, nil
}

// SetPtrAttribute sets attributes on a previously allocated memory region.
// The supported attributes are:
//		SynncMemOpsAttr:
//			A boolean attribute that can either be set (1) or unset (0).
//			When set, the region of memory that ptr points to is guaranteed to always synchronize memory operations that are synchronous.
//			If there are some previously initiated synchronous memory operations that are pending when this attribute is set, the function does not return until those memory operations are complete.
//			See further documentation in the section titled "API synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.
// 			`value` will be considered as a pointer to an unsigned integer to which this attribute is to be set.
func (d DevicePtr) SetPtrAttribute(value unsafe.Pointer, attr PointerAttribute) error {
	devPtr := C.CUdeviceptr(d)
	a := C.CUpointer_attribute(attr)
	return result(C.cuPointerSetAttribute(value, a, devPtr))
}

// TODO: MemRange attributes
