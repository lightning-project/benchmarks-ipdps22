use std::alloc::{GlobalAlloc, Layout, System};

pub struct Allocator;

// There seem to be some issue with the default allocator combined with MPI.
// For now, we forward all allocations through the system allocator since
// this seems to resolve the issue.
unsafe impl GlobalAlloc for Allocator {
    #[track_caller]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }

    #[track_caller]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}
