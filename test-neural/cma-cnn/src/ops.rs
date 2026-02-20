//! # CNN Operations
//!
//! Implementation of fundamental operations for CNNs:
//! - im2col / col2im for efficient convolution
//! - Padding modes

use crate::Float;
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use ndarray::Axis;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::tensor::{Tensor4D, TensorShape};

/// Mode de padding pour les convolutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Padding {
    /// No padding (smaller output)
    Valid,
    /// Padding to preserve size (output = input)
    Same,
    /// Explicit padding
    Fixed(usize),
}

impl Padding {
    /// Computes the required padding
    pub fn compute(&self, kernel_size: usize) -> usize {
        match self {
            Padding::Valid => 0,
            Padding::Same => kernel_size / 2,
            Padding::Fixed(p) => *p,
        }
    }
}

/// Im2Col: Transforms an image into a column matrix for efficient convolution
///
/// # Principe (LeCun et al., 1998)
///
/// Instead of using nested loops for convolution, we reorganize
/// the image patches into columns, then perform a matrix multiplication.
///
/// For a 4x4 image with 2x2 kernel:
/// ```text
/// Image:          Patches → Columns:
/// [1 2 3 4]       [1 2]  [2 3]  [3 4]      [1 2 5 6]
/// [5 6 7 8]   →   [5 6], [6 7], [7 8], ... [2 3 6 7]  ← each column = flattened patch
/// [9 ...  ]                                [...]
/// ```
///
/// Convolution = Weights × Im2Col(Input)
///
/// # Arguments
/// * `input` - Tensor [batch, channels, height, width]
/// * `kernel_size` - Kernel size (square)
/// * `stride` - Step size
/// * `padding` - Padding (surrounding zeros)
/// * `batch_idx` - Batch index to process
///
/// # Returns
/// Matrix [kernel_size² × channels, out_height × out_width]
pub fn im2col_single(
    input: &Tensor4D,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    batch_idx: usize,
) -> Array2<Float> {
    let shape = input.shape();
    let data = input.data();

    // Output dimensions
    let out_h = (shape.height + 2 * padding - kernel_size) / stride + 1;
    let out_w = (shape.width + 2 * padding - kernel_size) / stride + 1;

    // Column size = kernel² × channels
    let col_size = kernel_size * kernel_size * shape.channels;
    // Number of columns = spatial positions
    let num_cols = out_h * out_w;

    // Pre-allocate with exact capacity
    let mut cols = Array2::zeros((col_size, num_cols));
    
    // Cache-aware optimization: reorder loops for contiguous memory access
    // The channel -> kh -> kw order is fixed by the output format
    // But we optimize data access by traversing per image row
    
    if padding == 0 {
        // No-padding case - optimized direct access
        // Traversal by (channel, kernel_row) for better cache locality
        for c in 0..shape.channels {
            let c_offset = c * kernel_size * kernel_size;
            for kh in 0..kernel_size {
                let row_base = c_offset + kh * kernel_size;
                for kw in 0..kernel_size {
                    let row_idx = row_base + kw;
                    
                    // Traverse output positions
                    let mut col_idx = 0;
                    for oh in 0..out_h {
                        let ih = oh * stride + kh;
                        for ow in 0..out_w {
                            let iw = ow * stride + kw;
                            cols[[row_idx, col_idx]] = data[[batch_idx, c, ih, iw]];
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    } else {
        // Padding case - bounds checking required
        let h_max = shape.height + padding;
        let w_max = shape.width + padding;
        
        for c in 0..shape.channels {
            let c_offset = c * kernel_size * kernel_size;
            for kh in 0..kernel_size {
                let row_base = c_offset + kh * kernel_size;
                for kw in 0..kernel_size {
                    let row_idx = row_base + kw;
                    
                    let mut col_idx = 0;
                    for oh in 0..out_h {
                        let ih = oh * stride + kh;
                        let ih_valid = ih >= padding && ih < h_max;
                        let ih_real = ih.wrapping_sub(padding); // Safe car on check ih_valid
                        
                        for ow in 0..out_w {
                            let iw = ow * stride + kw;
                            
                            let val = if ih_valid && iw >= padding && iw < w_max {
                                data[[batch_idx, c, ih_real, iw - padding]]
                            } else {
                                0.0
                            };
                            
                            cols[[row_idx, col_idx]] = val;
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    cols
}

/// Im2Col for the entire batch (returns Vec of matrices for each image)
#[allow(dead_code)]
pub fn im2col(
    input: &Tensor4D,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<Array2<Float>> {
    let shape = input.shape();
    (0..shape.batch)
        .map(|b| im2col_single(input, kernel_size, stride, padding, b))
        .collect()
}

/// Col2Im: Inverse of im2col, used for the backward pass
///
/// Reconstructs the input gradient from the column gradient.
pub fn col2im(
    cols: &Array2<Float>,
    original_shape: TensorShape,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor4D {
    let out_h = (original_shape.height + 2 * padding - kernel_size) / stride + 1;
    let out_w = (original_shape.width + 2 * padding - kernel_size) / stride + 1;

    let padded_h = original_shape.height + 2 * padding;
    let padded_w = original_shape.width + 2 * padding;

    let mut data = Array4::zeros((
        original_shape.batch.min(1),
        original_shape.channels,
        padded_h,
        padded_w,
    ));

    for b in 0..original_shape.batch.min(1) {
        let mut col_idx = 0;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut row_idx = 0;

                for c in 0..original_shape.channels {
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;

                            // Accumulate (multiple patches may overlap)
                            data[[b, c, ih, iw]] += cols[[row_idx, col_idx]];
                            row_idx += 1;
                        }
                    }
                }
                col_idx += 1;
            }
        }
    }

    // Remove padding if necessary
    if padding > 0 {
        let mut result = Array4::zeros((
            original_shape.batch.min(1),
            original_shape.channels,
            original_shape.height,
            original_shape.width,
        ));

        for b in 0..original_shape.batch.min(1) {
            for c in 0..original_shape.channels {
                for h in 0..original_shape.height {
                    for w in 0..original_shape.width {
                        result[[b, c, h, w]] = data[[b, c, h + padding, w + padding]];
                    }
                }
            }
        }

        Tensor4D::from_array(result)
    } else {
        Tensor4D::from_array(data)
    }
}

/// Naive 2D Convolution (reference implementation for tests)
///
/// Direct implementation with nested loops - O(B × Co × Ci × H × W × K²).
/// Used only to validate that `conv2d_im2col` produces the same results.
///
/// **Note**: Do not use in production, prefer `conv2d_im2col`.
#[cfg(test)]
pub fn conv2d_naive(
    input: &Tensor4D,
    weights: &Array4<Float>, // [out_channels, in_channels, kH, kW]
    bias: Option<&ndarray::Array1<Float>>,
    stride: usize,
    padding: usize,
) -> Tensor4D {
    let in_shape = input.shape();
    let in_data = input.data();

    let out_channels = weights.dim().0;
    let kernel_h = weights.dim().2;
    let kernel_w = weights.dim().3;

    let out_h = (in_shape.height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (in_shape.width + 2 * padding - kernel_w) / stride + 1;

    let mut output = Array4::zeros((in_shape.batch, out_channels, out_h, out_w));

    for b in 0..in_shape.batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;

                    for ic in 0..in_shape.channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = (oh * stride + kh) as i32 - padding as i32;
                                let iw = (ow * stride + kw) as i32 - padding as i32;

                                if ih >= 0
                                    && ih < in_shape.height as i32
                                    && iw >= 0
                                    && iw < in_shape.width as i32
                                {
                                    sum += in_data[[b, ic, ih as usize, iw as usize]]
                                        * weights[[oc, ic, kh, kw]];
                                }
                            }
                        }
                    }

                    // Add bias
                    if let Some(b_vec) = bias {
                        sum += b_vec[oc];
                    }

                    output[[b, oc, oh, ow]] = sum;
                }
            }
        }
    }

    Tensor4D::from_array(output)
}

/// Optimized 2D Convolution with im2col + matrix multiplication (GEMM)
///
/// This implementation is ~10-100x faster than conv2d_naive because it
/// transforms the convolution into a single matrix multiplication.
///
/// # Principle
/// 1. Transform the input into columns with im2col: [K²×C, H'×W']
/// 2. Reshape weights into a matrix: [Out_C, K²×C]  
/// 3. Multiply: Output = Weights × Im2Col(Input)
/// 4. Reshape the result to [Batch, Out_C, H', W']
pub fn conv2d_im2col(
    input: &Tensor4D,
    weights: &Array4<Float>, // [out_channels, in_channels, kH, kW]
    bias: Option<&ndarray::Array1<Float>>,
    stride: usize,
    padding: usize,
) -> Tensor4D {
    #[cfg(feature = "parallel")]
    return conv2d_im2col_parallel(input, weights, bias, stride, padding);

    #[cfg(not(feature = "parallel"))]
    conv2d_im2col_sequential(input, weights, bias, stride, padding)
}

/// Sequential version of conv2d_im2col
#[allow(dead_code)]
fn conv2d_im2col_sequential(
    input: &Tensor4D,
    weights: &Array4<Float>,
    bias: Option<&ndarray::Array1<Float>>,
    stride: usize,
    padding: usize,
) -> Tensor4D {
    let in_shape = input.shape();
    let out_channels = weights.dim().0;
    let in_channels = weights.dim().1;
    let kernel_h = weights.dim().2;
    let kernel_w = weights.dim().3;

    let out_h = (in_shape.height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (in_shape.width + 2 * padding - kernel_w) / stride + 1;

    // Reshape weights: [out_channels, in_channels, kH, kW] → [out_channels, in_channels * kH * kW]
    let weight_rows = out_channels;
    let weight_cols = in_channels * kernel_h * kernel_w;
    let weights_2d = weights
        .view()
        .into_shape_with_order((weight_rows, weight_cols))
        .expect("Failed to reshape weights");

    let mut output = Array4::zeros((in_shape.batch, out_channels, out_h, out_w));

    // Process each image in the batch
    for b in 0..in_shape.batch {
        // Im2col pour cette image: [K²×C, H'×W']
        let cols = im2col_single(input, kernel_h, stride, padding, b);

        // GEMM: [out_channels, K²×C] × [K²×C, H'×W'] = [out_channels, H'×W']
        let conv_result = weights_2d.dot(&cols);

        // Optimization: vectorized copy with direct reshape
        for oc in 0..out_channels {
            let bias_val = bias.map_or(0.0, |b_arr| b_arr[oc]);
            let row = conv_result.row(oc);
            let mut out_slice = output.slice_mut(ndarray::s![b, oc, .., ..]);
            
            // Direct copy with bias addition
            for (i, out_val) in out_slice.iter_mut().enumerate() {
                *out_val = row[i] + bias_val;
            }
        }
    }

    Tensor4D::from_array(output)
}

/// Parallelized version of conv2d_im2col with Rayon
#[cfg(feature = "parallel")]
fn conv2d_im2col_parallel(
    input: &Tensor4D,
    weights: &Array4<Float>,
    bias: Option<&ndarray::Array1<Float>>,
    stride: usize,
    padding: usize,
) -> Tensor4D {
    let in_shape = input.shape();
    let out_channels = weights.dim().0;
    let in_channels = weights.dim().1;
    let kernel_h = weights.dim().2;
    let kernel_w = weights.dim().3;

    let out_h = (in_shape.height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (in_shape.width + 2 * padding - kernel_w) / stride + 1;

    // Reshape weights
    let weight_rows = out_channels;
    let weight_cols = in_channels * kernel_h * kernel_w;
    let weights_2d = weights
        .view()
        .into_shape_with_order((weight_rows, weight_cols))
        .expect("Failed to reshape weights");

    // Process each image in the batch in parallel
    let batch_results: Vec<Array4<Float>> = (0..in_shape.batch)
        .into_par_iter()
        .map(|b| {
            let cols = im2col_single(input, kernel_h, stride, padding, b);
            let conv_result = weights_2d.dot(&cols);

            let mut batch_output = Array4::zeros((1, out_channels, out_h, out_w));
            
            // Optimization: vectorized copy per channel
            for oc in 0..out_channels {
                let bias_val = bias.map_or(0.0, |bias_arr| bias_arr[oc]);
                let row = conv_result.row(oc);
                let mut out_slice = batch_output.slice_mut(ndarray::s![0, oc, .., ..]);
                
                for (i, out_val) in out_slice.iter_mut().enumerate() {
                    *out_val = row[i] + bias_val;
                }
            }
            batch_output
        })
        .collect();

    // Concatenate results
    let views: Vec<_> = batch_results.iter().map(|a| a.view()).collect();
    let output =
        ndarray::concatenate(Axis(0), &views).expect("Failed to concatenate batch results");

    Tensor4D::from_array(output)
}

/// Max Pooling 2D
pub fn maxpool2d(input: &Tensor4D, pool_size: usize, stride: usize) -> (Tensor4D, Array4<usize>) {
    #[cfg(feature = "parallel")]
    return maxpool2d_parallel(input, pool_size, stride);
    
    #[cfg(not(feature = "parallel"))]
    maxpool2d_sequential(input, pool_size, stride)
}

/// Sequential version of maxpool2d
#[allow(dead_code)]
fn maxpool2d_sequential(input: &Tensor4D, pool_size: usize, stride: usize) -> (Tensor4D, Array4<usize>) {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;

    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    // Store max indices for backward pass
    let mut indices = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    
    for b in 0..shape.batch {
        for c in 0..shape.channels {
            for oh in 0..out_h {
                let ih_base = oh * stride;
                for ow in 0..out_w {
                    let iw_base = ow * stride;
                    
                    // Optimization: first value as initial
                    let mut max_val = data[[b, c, ih_base, iw_base]];
                    let mut max_idx = 0usize;

                    // Linear traversal of the window
                    for ph in 0..pool_size {
                        let ih = ih_base + ph;
                        for pw in 0..pool_size {
                            let iw = iw_base + pw;
                            let val = data[[b, c, ih, iw]];

                            if val > max_val {
                                max_val = val;
                                max_idx = ph * pool_size + pw;
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                    indices[[b, c, oh, ow]] = max_idx;
                }
            }
        }
    }

    (Tensor4D::from_array(output), indices)
}

/// Parallel version of maxpool2d - parallelizes over batch × channels
#[cfg(feature = "parallel")]
fn maxpool2d_parallel(input: &Tensor4D, pool_size: usize, stride: usize) -> (Tensor4D, Array4<usize>) {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;

    // Parallelize over (batch, channel) pairs
    let results: Vec<(usize, usize, Vec<Float>, Vec<usize>)> = (0..shape.batch)
        .into_par_iter()
        .flat_map(|b| {
            (0..shape.channels).into_par_iter().map(move |c| {
                let mut out_vals = Vec::with_capacity(out_h * out_w);
                let mut out_idxs = Vec::with_capacity(out_h * out_w);
                
                for oh in 0..out_h {
                    let ih_base = oh * stride;
                    for ow in 0..out_w {
                        let iw_base = ow * stride;
                        
                        let mut max_val = data[[b, c, ih_base, iw_base]];
                        let mut max_idx = 0usize;

                        for ph in 0..pool_size {
                            let ih = ih_base + ph;
                            for pw in 0..pool_size {
                                let iw = iw_base + pw;
                                let val = data[[b, c, ih, iw]];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = ph * pool_size + pw;
                                }
                            }
                        }
                        out_vals.push(max_val);
                        out_idxs.push(max_idx);
                    }
                }
                (b, c, out_vals, out_idxs)
            })
        })
        .collect();

    // Rebuild arrays
    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    let mut indices = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    
    for (b, c, vals, idxs) in results {
        for (i, (val, idx)) in vals.into_iter().zip(idxs.into_iter()).enumerate() {
            let oh = i / out_w;
            let ow = i % out_w;
            output[[b, c, oh, ow]] = val;
            indices[[b, c, oh, ow]] = idx;
        }
    }

    (Tensor4D::from_array(output), indices)
}

/// Average Pooling 2D
pub fn avgpool2d(input: &Tensor4D, pool_size: usize, stride: usize) -> Tensor4D {
    #[cfg(feature = "parallel")]
    return avgpool2d_parallel(input, pool_size, stride);
    
    #[cfg(not(feature = "parallel"))]
    avgpool2d_sequential(input, pool_size, stride)
}

/// Sequential version of avgpool2d
#[allow(dead_code)]
fn avgpool2d_sequential(input: &Tensor4D, pool_size: usize, stride: usize) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;
    // Precompute divisor (once)
    let pool_area_inv = 1.0 / (pool_size * pool_size) as Float;

    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));

    for b in 0..shape.batch {
        for c in 0..shape.channels {
            for oh in 0..out_h {
                let ih_base = oh * stride;
                for ow in 0..out_w {
                    let iw_base = ow * stride;
                    let mut sum = 0.0;

                    // Optimized loop with precomputed base
                    for ph in 0..pool_size {
                        let ih = ih_base + ph;
                        for pw in 0..pool_size {
                            let iw = iw_base + pw;
                            sum += data[[b, c, ih, iw]];
                        }
                    }

                    // Multiplication instead of division (faster)
                    output[[b, c, oh, ow]] = sum * pool_area_inv;
                }
            }
        }
    }

    Tensor4D::from_array(output)
}

/// Parallel version of avgpool2d
#[cfg(feature = "parallel")]
fn avgpool2d_parallel(input: &Tensor4D, pool_size: usize, stride: usize) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;
    let pool_area_inv = 1.0 / (pool_size * pool_size) as Float;

    // Parallelize over (batch, channel) pairs
    let results: Vec<(usize, usize, Vec<Float>)> = (0..shape.batch)
        .into_par_iter()
        .flat_map(|b| {
            (0..shape.channels).into_par_iter().map(move |c| {
                let mut out_vals = Vec::with_capacity(out_h * out_w);
                
                for oh in 0..out_h {
                    let ih_base = oh * stride;
                    for ow in 0..out_w {
                        let iw_base = ow * stride;
                        let mut sum = 0.0;

                        for ph in 0..pool_size {
                            let ih = ih_base + ph;
                            for pw in 0..pool_size {
                                let iw = iw_base + pw;
                                sum += data[[b, c, ih, iw]];
                            }
                        }
                        out_vals.push(sum * pool_area_inv);
                    }
                }
                (b, c, out_vals)
            })
        })
        .collect();

    // Rebuild array
    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    for (b, c, vals) in results {
        for (i, val) in vals.into_iter().enumerate() {
            let oh = i / out_w;
            let ow = i % out_w;
            output[[b, c, oh, ow]] = val;
        }
    }

    Tensor4D::from_array(output)
}

/// Global Average Pooling 2D
///
/// Reduces [batch, channels, H, W] → [batch, channels, 1, 1]
/// Used in modern architectures (ResNet, EfficientNet)
pub fn global_avgpool2d(input: &Tensor4D) -> Tensor4D {
    #[cfg(feature = "parallel")]
    return global_avgpool2d_parallel(input);
    
    #[cfg(not(feature = "parallel"))]
    global_avgpool2d_sequential(input)
}

/// Sequential version of global_avgpool2d
#[allow(dead_code)]
fn global_avgpool2d_sequential(input: &Tensor4D) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();
    // Precompute: multiplication instead of division
    let spatial_size_inv = 1.0 / (shape.height * shape.width) as Float;

    let mut output = Array4::zeros((shape.batch, shape.channels, 1, 1));

    for b in 0..shape.batch {
        for c in 0..shape.channels {
            // Optimization: uses ndarray slicing + iter().sum()
            let channel_slice = data.slice(ndarray::s![b, c, .., ..]);
            let sum: Float = channel_slice.iter().sum();
            output[[b, c, 0, 0]] = sum * spatial_size_inv;
        }
    }

    Tensor4D::from_array(output)
}

/// Parallel version of global_avgpool2d
#[cfg(feature = "parallel")]
fn global_avgpool2d_parallel(input: &Tensor4D) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();
    let spatial_size_inv = 1.0 / (shape.height * shape.width) as Float;

    // Parallelize over (batch, channel)
    let results: Vec<(usize, usize, Float)> = (0..shape.batch)
        .into_par_iter()
        .flat_map(|b| {
            (0..shape.channels).into_par_iter().map(move |c| {
                let channel_slice = data.slice(ndarray::s![b, c, .., ..]);
                let sum: Float = channel_slice.iter().sum();
                (b, c, sum * spatial_size_inv)
            })
        })
        .collect();

    let mut output = Array4::zeros((shape.batch, shape.channels, 1, 1));
    for (b, c, val) in results {
        output[[b, c, 0, 0]] = val;
    }

    Tensor4D::from_array(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding_modes() {
        assert_eq!(Padding::Valid.compute(3), 0);
        assert_eq!(Padding::Same.compute(3), 1);
        assert_eq!(Padding::Same.compute(5), 2);
        assert_eq!(Padding::Fixed(2).compute(3), 2);
    }

    #[test]
    fn test_maxpool2d() {
        // Input 1x1x4x4
        let data = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let input = Tensor4D::from_array(data);

        let (output, _indices) = maxpool2d(&input, 2, 2);
        let out_data = output.data();

        // Pool 2x2 stride 2 → 2x2 output
        assert_eq!(output.shape().height, 2);
        assert_eq!(output.shape().width, 2);
        // Max values
        assert_eq!(out_data[[0, 0, 0, 0]], 6.0);
        assert_eq!(out_data[[0, 0, 0, 1]], 8.0);
        assert_eq!(out_data[[0, 0, 1, 0]], 14.0);
        assert_eq!(out_data[[0, 0, 1, 1]], 16.0);
    }

    #[test]
    fn test_avgpool2d() {
        let data = Array4::from_shape_vec(
            (1, 1, 4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let input = Tensor4D::from_array(data);

        let output = avgpool2d(&input, 2, 2);
        let out_data = output.data();

        // Avg of [1,2,5,6] = 3.5
        assert_eq!(out_data[[0, 0, 0, 0]], 3.5);
        // Avg of [3,4,7,8] = 5.5
        assert_eq!(out_data[[0, 0, 0, 1]], 5.5);
    }

    #[test]
    fn test_global_avgpool() {
        let data = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, // Channel 0: avg = 2.5
                5.0, 6.0, 7.0, 8.0, // Channel 1: avg = 6.5
            ],
        )
        .unwrap();
        let input = Tensor4D::from_array(data);

        let output = global_avgpool2d(&input);
        let out_data = output.data();

        assert_eq!(output.shape().height, 1);
        assert_eq!(output.shape().width, 1);
        assert_eq!(out_data[[0, 0, 0, 0]], 2.5);
        assert_eq!(out_data[[0, 1, 0, 0]], 6.5);
    }

    #[test]
    fn test_conv2d_naive() {
        // Input: 1x1x3x3
        let input_data = Array4::from_shape_vec(
            (1, 1, 3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let input = Tensor4D::from_array(input_data);

        // Kernel: 1x1x2x2 (identity-like)
        let weights = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let output = conv2d_naive(&input, &weights, None, 1, 0);
        let out_data = output.data();

        // Output: 1x1x2x2
        assert_eq!(output.shape().height, 2);
        assert_eq!(output.shape().width, 2);
        // With this kernel, copies the top-left corner of each patch
        assert_eq!(out_data[[0, 0, 0, 0]], 1.0);
        assert_eq!(out_data[[0, 0, 0, 1]], 2.0);
        assert_eq!(out_data[[0, 0, 1, 0]], 4.0);
        assert_eq!(out_data[[0, 0, 1, 1]], 5.0);
    }
}
