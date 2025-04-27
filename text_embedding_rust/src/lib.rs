mod models;
use std::{
    ffi::{c_char, CStr, CString},
    ptr,
};

use models::text_embedder::{BertModelTypes, TextEmbedder};

// --- FFI wrappers ---

/// Opaque handle
pub type TextEmbedderHandle = *mut TextEmbedder;

/// Create a new TextEmbedder and return a raw pointer
#[no_mangle]
pub extern "C" fn text_embedder_create(
    model_id: *const c_char,
    out_error: *mut *mut c_char,
) -> TextEmbedderHandle {
    let cstr = unsafe { CStr::from_ptr(model_id) };
    let rust_str = cstr.to_str().unwrap_or("");
    match TextEmbedder::new(rust_str) {
        Ok(embedder) => Box::into_raw(Box::new(embedder)),
        Err(e) => {
            let msg = format!("Failed to load model: {}", e);
            let c_msg = CString::new(msg).unwrap();
            unsafe {
                *out_error = c_msg.into_raw();
            }
            ptr::null_mut()
        }
    }
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

#[no_mangle]
pub extern "C" fn text_embedder_embed_batch(
    h: TextEmbedderHandle,
    prompts: *const *const c_char,
    n_prompts: usize,
    out_total_len: *mut usize,
) -> *mut f32 {
    let t = unsafe {
        assert!(!h.is_null());
        &*h
    };

    if prompts.is_null() || out_total_len.is_null() {
        return std::ptr::null_mut();
    }

    let c_prompts: &[*const c_char] = unsafe { std::slice::from_raw_parts(prompts, n_prompts) };
    let mut rust_prompts = Vec::with_capacity(n_prompts);
    for &p in c_prompts {
        if p.is_null() {
            return std::ptr::null_mut();
        }
        let cstr = unsafe { CStr::from_ptr(p) };
        match cstr.to_str() {
            Ok(s) => rust_prompts.push(s),
            Err(_) => return std::ptr::null_mut(),
        }
    }

    let embeddings = match t.embed_batch(&rust_prompts) {
        Ok(e) => e,
        Err(_) => return std::ptr::null_mut(),
    };

    if embeddings.is_empty() {
        return std::ptr::null_mut();
    }

    let embedding_dim = embeddings[0].len();
    let total_len = n_prompts * embedding_dim;

    let mut flat_embeddings = Vec::with_capacity(total_len);
    for emb in embeddings {
        flat_embeddings.extend_from_slice(&emb);
    }

    let ptr = flat_embeddings.as_mut_ptr();
    std::mem::forget(flat_embeddings); // transfer ownership to caller

    unsafe {
        *out_total_len = total_len;
    }

    ptr
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

#[no_mangle]
pub extern "C" fn text_embedder_free_string(str: *mut c_char) {
    if !str.is_null() {
        unsafe {
            drop(CString::from_raw(str));
        }
    }
}

#[no_mangle]
pub extern "C" fn text_embedder_list_models() -> *mut c_char {
    let c_msg = CString::new(BertModelTypes::all_hf_hub_ids_as_string()).unwrap();
    c_msg.into_raw()
}
