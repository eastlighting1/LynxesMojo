use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::{GFError, Result};

static CONFIGURED_LIB: OnceLock<PathBuf> = OnceLock::new();

pub fn configure_mojo_runtime(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref().to_path_buf();
    if let Some(existing) = CONFIGURED_LIB.get() {
        if existing == &path {
            return Ok(());
        }
        return Err(GFError::InvalidConfig {
            message: format!(
                "Mojo runtime already configured at {}, cannot reset to {}",
                existing.display(),
                path.display()
            ),
        });
    }
    CONFIGURED_LIB
        .set(path)
        .map_err(|_| GFError::InvalidConfig {
            message: "failed to configure Mojo runtime path".to_owned(),
        })
}

pub fn mojo_runtime_path() -> Option<PathBuf> {
    configured_or_env_path()
}

#[allow(dead_code)]
pub(crate) struct StructuralDegreeInputs<'a> {
    pub node_to_edge_idx: &'a [i64],
    pub out_offsets: &'a [u32],
    pub out_edge_ids: &'a [u32],
    pub in_offsets: &'a [u32],
    pub in_edge_ids: &'a [u32],
    pub edge_allowed: &'a [u8],
}

pub(crate) fn compute_structural_degrees(
    inputs: StructuralDegreeInputs<'_>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    if inputs.out_edge_ids.len() != inputs.edge_allowed.len()
        || inputs.in_edge_ids.len() != inputs.edge_allowed.len()
    {
        return Err(GFError::LengthMismatch {
            expected: inputs.out_edge_ids.len(),
            actual: inputs.edge_allowed.len(),
        });
    }

    platform::compute_structural_degrees(inputs)
}

fn configured_or_env_path() -> Option<PathBuf> {
    CONFIGURED_LIB
        .get()
        .cloned()
        .or_else(|| std::env::var_os("LYNXES_MOJO_LIB").map(PathBuf::from))
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{configured_or_env_path, StructuralDegreeInputs};
    use crate::{GFError, Result};
    use libloading::Library;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    type StructuralDegreesFn = unsafe extern "C" fn(
        usize,
        *const i64,
        *const u32,
        *const u32,
        *const u32,
        *const u32,
        *const u8,
        *mut i64,
        *mut i64,
        *mut i64,
    ) -> i32;

    struct MojoRuntime {
        _lib: Library,
        structural_degrees: StructuralDegreesFn,
    }

    static RUNTIME: OnceLock<MojoRuntime> = OnceLock::new();

    pub(super) fn compute_structural_degrees(
        inputs: StructuralDegreeInputs<'_>,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let runtime = runtime()?;
        let node_count = inputs.node_to_edge_idx.len();
        let mut out_degree = vec![0i64; node_count];
        let mut in_degree = vec![0i64; node_count];
        let mut total_degree = vec![0i64; node_count];

        let status = unsafe {
            (runtime.structural_degrees)(
                node_count,
                inputs.node_to_edge_idx.as_ptr(),
                inputs.out_offsets.as_ptr(),
                inputs.out_edge_ids.as_ptr(),
                inputs.in_offsets.as_ptr(),
                inputs.in_edge_ids.as_ptr(),
                inputs.edge_allowed.as_ptr(),
                out_degree.as_mut_ptr(),
                in_degree.as_mut_ptr(),
                total_degree.as_mut_ptr(),
            )
        };

        if status != 0 {
            return Err(GFError::UnsupportedOperation {
                message: format!("Mojo structural degree kernel failed with status {status}"),
            });
        }

        Ok((out_degree, in_degree, total_degree))
    }

    fn runtime() -> Result<&'static MojoRuntime> {
        if let Some(runtime) = RUNTIME.get() {
            return Ok(runtime);
        }
        let runtime = load_runtime().map_err(|message| GFError::UnsupportedOperation {
            message: format!("Mojo runtime is required for this operation: {message}"),
        })?;
        let _ = RUNTIME.set(runtime);
        Ok(RUNTIME
            .get()
            .expect("Mojo runtime was just initialized successfully"))
    }

    fn load_runtime() -> std::result::Result<MojoRuntime, String> {
        let path = configured_or_env_path().ok_or_else(|| {
            "set LYNXES_MOJO_LIB or call configure_mojo_runtime() with liblynxes_mojo_kernels.so"
                .to_owned()
        })?;
        unsafe { load_runtime_from_path(path) }
    }

    unsafe fn load_runtime_from_path(path: PathBuf) -> std::result::Result<MojoRuntime, String> {
        let lib = Library::new(&path)
            .map_err(|err| format!("failed to load {}: {err}", path.display()))?;
        let structural_degrees = {
            let symbol = lib
                .get::<StructuralDegreesFn>(b"lynxes_structural_degrees_i64\0")
                .map_err(|err| {
                    format!(
                        "failed to load symbol lynxes_structural_degrees_i64 from {}: {err}",
                        path.display()
                    )
                })?;
            *symbol
        };
        Ok(MojoRuntime {
            _lib: lib,
            structural_degrees,
        })
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::StructuralDegreeInputs;
    use crate::{GFError, Result};

    pub(super) fn compute_structural_degrees(
        _inputs: StructuralDegreeInputs<'_>,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        Err(GFError::UnsupportedOperation {
            message: "Mojo structural feature kernels are only supported by Linux builds"
                .to_owned(),
        })
    }
}
