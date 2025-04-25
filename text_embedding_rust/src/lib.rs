mod models;
use std::ffi::{c_char, CStr};

use models::text_embedder::TextEmbedder;

// --- FFI wrappers ---

/// Opaque handle
pub type TextEmbedderHandle = *mut TextEmbedder;

/// Create a new TextEmbedder and return a raw pointer
#[no_mangle]
pub extern "C" fn text_embedder_create(model_id: *const c_char) -> TextEmbedderHandle {
    let cstr = unsafe { CStr::from_ptr(model_id) };
    let rust_str = cstr.to_str().unwrap_or("");
    let t = Box::new(TextEmbedder::new(rust_str).expect("Error loading model!"));
    Box::into_raw(t)
}

/// Drop it
#[no_mangle]
pub extern "C" fn text_embedder_free(h: TextEmbedderHandle) {
    if !h.is_null() {
        unsafe {
            let _ = Box::from_raw(h);
        } // dropped here
    }
}

/// Embed; returns a newly-malloc’d float array and writes its length
#[no_mangle]
pub extern "C" fn text_embedder_embed(
    h: TextEmbedderHandle,
    prompt: *const c_char,
    out_len: *mut usize,
) -> *mut f32 {
    let t = unsafe {
        assert!(!h.is_null());
        &*h
    };
    let cstr = unsafe { CStr::from_ptr(prompt) };
    let rust_str = cstr.to_str().unwrap_or("");
    match t.embed(rust_str) {
        Ok(mut v) => {
            let len = v.len();
            let ptr = v.as_mut_ptr();
            std::mem::forget(v); // ownership → caller
            unsafe {
                *out_len = len;
            }
            ptr
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free the float array returned by `embed`
#[no_mangle]
pub extern "C" fn text_embedder_free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        // reconstruct and drop
        unsafe {
            Vec::from_raw_parts(ptr, len, len);
        }
    }
}

/// Get the output dimension
#[no_mangle]
pub extern "C" fn text_embedder_output_dims(h: TextEmbedderHandle) -> usize {
    let t = unsafe {
        assert!(!h.is_null());
        &*h
    };
    t.get_output_dims()
}
