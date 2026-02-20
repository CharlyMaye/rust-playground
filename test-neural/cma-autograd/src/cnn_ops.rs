//! # Shared CNN Operations
//!
//! Bridge between autograd tensors (`ArrayD<Float>`) and cma-cnn's optimized ops
//! (`Tensor4D` / `Array4<Float>`).
//!
//! Instead of reimplementing im2col, maxpool, avgpool, etc. with naive loops,
//! autograd delegates the forward computation to cma-cnn's cache-optimized,
//! optionally parallelized implementations.
//!
//! ## Conversion Strategy
//!
//! - Autograd works with `ArrayD<Float>` (dynamic dimensions)
//! - cma-cnn works with `Tensor4D` wrapping `Array4<Float>` (fixed 4D)
//! - This module provides zero-copy-when-possible conversion at the boundary

use crate::Float;
use cma_cnn::tensor::Tensor4D;
use ndarray::{Array2, Array4, ArrayD, IxDyn};

// ═══════════════════════════════════════════════════════════════════════════
// Type Conversions
// ═══════════════════════════════════════════════════════════════════════════

/// Convert an `ArrayD<Float>` to `Array4<Float>`.
///
/// Panics if the input is not exactly 4-dimensional.
#[inline]
pub fn arrayd_to_array4(input: &ArrayD<Float>) -> Array4<Float> {
    input
        .clone()
        .into_dimensionality::<ndarray::Ix4>()
        .expect("cnn_ops: input must be 4D [N, C, H, W]")
}

/// Convert an `ArrayD<Float>` to `Tensor4D` (cma-cnn format).
///
/// Panics if the input is not exactly 4-dimensional.
#[inline]
pub fn arrayd_to_tensor4d(input: &ArrayD<Float>) -> Tensor4D {
    Tensor4D::from_array(arrayd_to_array4(input))
}

/// Convert a `Tensor4D` result back to `ArrayD<Float>`.
#[inline]
pub fn tensor4d_to_arrayd(output: &Tensor4D) -> ArrayD<Float> {
    output.data().clone().into_dyn()
}

/// Convert an `Array4<Float>` to `ArrayD<Float>`.
#[inline]
pub fn array4_to_arrayd(output: &Array4<Float>) -> ArrayD<Float> {
    output.clone().into_dyn()
}

// ═══════════════════════════════════════════════════════════════════════════
// im2col — delegates to cma-cnn's cache-optimized implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Optimized im2col using cma-cnn's cache-aware implementation.
///
/// Returns columns in autograd's expected format: `[N*OH*OW, C_in*kH*kW]`
/// (each row is one flattened patch, suitable for `col @ weight^T`).
///
/// cma-cnn's `im2col_single` returns `[kH*kW*C_in, OH*OW]` (transposed),
/// so we transpose per-batch and stack.
pub fn im2col_optimized(
    input: &ArrayD<Float>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> ArrayD<Float> {
    let shape = input.shape();
    let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;
    let col_size = channels * kernel_size * kernel_size;
    let spatial = out_h * out_w;
    let total_rows = batch * spatial;

    // Convert to Tensor4D for cma-cnn
    let tensor4d = arrayd_to_tensor4d(input);

    let mut col = Array2::<Float>::zeros((total_rows, col_size));

    for b in 0..batch {
        // cma-cnn returns [col_size, spatial] — transpose to [spatial, col_size]
        let cols_b: Array2<Float> =
            cma_cnn::im2col_single(&tensor4d, kernel_size, stride, padding, b);
        let cols_b_t = cols_b.t();

        // Copy transposed block into output
        let row_offset = b * spatial;
        col.slice_mut(ndarray::s![row_offset..row_offset + spatial, ..])
            .assign(&cols_b_t);
    }

    col.into_dyn()
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxPool2D — delegates to cma-cnn's optimized (optionally parallel) impl
// ═══════════════════════════════════════════════════════════════════════════

/// Result of max pooling, with indices in autograd-compatible format.
pub struct MaxPool2DResult {
    /// Pooled output as ArrayD [N, C, OH, OW]
    pub output: ArrayD<Float>,
    /// Absolute row indices of max values [N, C, OH, OW]
    pub max_indices_h: ndarray::ArrayD<usize>,
    /// Absolute column indices of max values [N, C, OH, OW]
    pub max_indices_w: ndarray::ArrayD<usize>,
}

/// Optimized MaxPool2D using cma-cnn's implementation.
///
/// Returns output + absolute (h, w) indices for autograd backward.
pub fn maxpool2d_optimized(
    input: &ArrayD<Float>,
    kernel_size: usize,
    stride: usize,
) -> MaxPool2DResult {
    let shape = input.shape();
    let (batch, channels, _h, _w) = (shape[0], shape[1], shape[2], shape[3]);

    let tensor4d = arrayd_to_tensor4d(input);
    let (out_tensor, flat_indices) = cma_cnn::maxpool2d(&tensor4d, kernel_size, stride);

    let out_shape = out_tensor.shape();
    let out_h = out_shape.height;
    let out_w = out_shape.width;

    // Convert flat_indices (ph*pool_size + pw) → absolute (h, w) positions
    let mut max_indices_h =
        ndarray::ArrayD::<usize>::zeros(IxDyn(&[batch, channels, out_h, out_w]));
    let mut max_indices_w =
        ndarray::ArrayD::<usize>::zeros(IxDyn(&[batch, channels, out_h, out_w]));

    for b in 0..batch {
        for c in 0..channels {
            for i in 0..out_h {
                for j in 0..out_w {
                    let flat_idx = flat_indices[[b, c, i, j]];
                    let ph = flat_idx / kernel_size;
                    let pw = flat_idx % kernel_size;
                    max_indices_h[[b, c, i, j]] = i * stride + ph;
                    max_indices_w[[b, c, i, j]] = j * stride + pw;
                }
            }
        }
    }

    MaxPool2DResult {
        output: tensor4d_to_arrayd(&out_tensor),
        max_indices_h,
        max_indices_w,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AvgPool2D — delegates to cma-cnn's optimized implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Optimized AvgPool2D using cma-cnn's implementation.
pub fn avgpool2d_optimized(
    input: &ArrayD<Float>,
    kernel_size: usize,
    stride: usize,
) -> ArrayD<Float> {
    let tensor4d = arrayd_to_tensor4d(input);
    let result = cma_cnn::avgpool2d(&tensor4d, kernel_size, stride);
    tensor4d_to_arrayd(&result)
}

// ═══════════════════════════════════════════════════════════════════════════
// GlobalAvgPool2D — delegates to cma-cnn's optimized implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Optimized GlobalAvgPool2D using cma-cnn's implementation.
pub fn global_avgpool2d_optimized(input: &ArrayD<Float>) -> ArrayD<Float> {
    let tensor4d = arrayd_to_tensor4d(input);
    let result = cma_cnn::global_avgpool2d(&tensor4d);
    tensor4d_to_arrayd(&result)
}
