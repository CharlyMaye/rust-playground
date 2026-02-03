//! # Opérations CNN
//!
//! Implémentation des opérations fondamentales pour les CNN:
//! - im2col / col2im pour convolution efficace
//! - Padding modes

use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};

use crate::tensor::{Tensor4D, TensorShape};

/// Mode de padding pour les convolutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Padding {
    /// Pas de padding (output plus petit)
    Valid,
    /// Padding pour conserver la taille (output = input)
    Same,
    /// Padding explicite
    Fixed(usize),
}

impl Padding {
    /// Calcule le padding nécessaire
    pub fn compute(&self, kernel_size: usize) -> usize {
        match self {
            Padding::Valid => 0,
            Padding::Same => kernel_size / 2,
            Padding::Fixed(p) => *p,
        }
    }
}

/// Im2Col: Transforme une image en matrice de colonnes pour convolution efficace
///
/// # Principe (LeCun et al., 1998)
///
/// Au lieu de faire des boucles nested pour la convolution, on réorganise
/// les patches de l'image en colonnes, puis on fait une multiplication matricielle.
///
/// Pour une image 4x4 avec kernel 2x2:
/// ```text
/// Image:          Patches → Colonnes:
/// [1 2 3 4]       [1 2]  [2 3]  [3 4]      [1 2 5 6]
/// [5 6 7 8]   →   [5 6], [6 7], [7 8], ... [2 3 6 7]  ← chaque colonne = patch aplati
/// [9 ...  ]                                [...]
/// ```
///
/// Convolution = Weights × Im2Col(Input)
///
/// # Arguments
/// * `input` - Tenseur [batch, channels, height, width]
/// * `kernel_size` - Taille du kernel (carré)
/// * `stride` - Pas de déplacement
/// * `padding` - Padding (zéros autour)
///
/// # Returns
/// Matrice [batch, kernel_size² × channels, out_height × out_width]
pub fn im2col(input: &Tensor4D, kernel_size: usize, stride: usize, padding: usize) -> Array2<f64> {
    let shape = input.shape();
    let data = input.data();

    // Dimensions de sortie
    let out_h = (shape.height + 2 * padding - kernel_size) / stride + 1;
    let out_w = (shape.width + 2 * padding - kernel_size) / stride + 1;

    // Taille d'une colonne = kernel² × channels
    let col_size = kernel_size * kernel_size * shape.channels;
    // Nombre de colonnes = positions spatiales
    let num_cols = out_h * out_w;

    // Pour simplifier, on traite batch_size = 1 pour l'instant
    // TODO: Support multi-batch
    let mut cols = Array2::zeros((col_size, num_cols));

    for b in 0..shape.batch.min(1) {
        // Premier élément du batch
        let mut col_idx = 0;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut row_idx = 0;

                for c in 0..shape.channels {
                    for kh in 0..kernel_size {
                        for kw in 0..kernel_size {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;

                            // Gestion du padding
                            let val = if ih >= padding
                                && ih < shape.height + padding
                                && iw >= padding
                                && iw < shape.width + padding
                            {
                                data[[b, c, ih - padding, iw - padding]]
                            } else {
                                0.0 // Zero-padding
                            };

                            cols[[row_idx, col_idx]] = val;
                            row_idx += 1;
                        }
                    }
                }
                col_idx += 1;
            }
        }
    }

    cols
}

/// Col2Im: Inverse de im2col, utilisé pour le backward pass
///
/// Reconstruit le gradient sur l'input à partir du gradient sur les colonnes.
pub fn col2im(
    cols: &Array2<f64>,
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

                            // Accumule (plusieurs patches peuvent se chevaucher)
                            data[[b, c, ih, iw]] += cols[[row_idx, col_idx]];
                            row_idx += 1;
                        }
                    }
                }
                col_idx += 1;
            }
        }
    }

    // Retire le padding si nécessaire
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

/// Convolution 2D naïve (pour référence et tests)
///
/// Implémentation directe avec boucles nested.
/// Plus lente que im2col mais plus lisible.
pub fn conv2d_naive(
    input: &Tensor4D,
    weights: &Array4<f64>, // [out_channels, in_channels, kH, kW]
    bias: Option<&ndarray::Array1<f64>>,
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

                    // Ajoute le biais
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

/// Max Pooling 2D
pub fn maxpool2d(input: &Tensor4D, pool_size: usize, stride: usize) -> (Tensor4D, Array4<usize>) {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;

    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));
    // Stocke les indices des max pour le backward
    let mut indices = Array4::zeros((shape.batch, shape.channels, out_h, out_w));

    for b in 0..shape.batch {
        for c in 0..shape.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx = 0;

                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;
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

/// Average Pooling 2D
pub fn avgpool2d(input: &Tensor4D, pool_size: usize, stride: usize) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();

    let out_h = (shape.height - pool_size) / stride + 1;
    let out_w = (shape.width - pool_size) / stride + 1;
    let pool_area = (pool_size * pool_size) as f64;

    let mut output = Array4::zeros((shape.batch, shape.channels, out_h, out_w));

    for b in 0..shape.batch {
        for c in 0..shape.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;

                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;
                            sum += data[[b, c, ih, iw]];
                        }
                    }

                    output[[b, c, oh, ow]] = sum / pool_area;
                }
            }
        }
    }

    Tensor4D::from_array(output)
}

/// Global Average Pooling 2D
///
/// Réduit [batch, channels, H, W] → [batch, channels, 1, 1]
/// Utilisé dans les architectures modernes (ResNet, EfficientNet)
pub fn global_avgpool2d(input: &Tensor4D) -> Tensor4D {
    let shape = input.shape();
    let data = input.data();
    let spatial_size = (shape.height * shape.width) as f64;

    let mut output = Array4::zeros((shape.batch, shape.channels, 1, 1));

    for b in 0..shape.batch {
        for c in 0..shape.channels {
            let mut sum = 0.0;
            for h in 0..shape.height {
                for w in 0..shape.width {
                    sum += data[[b, c, h, w]];
                }
            }
            output[[b, c, 0, 0]] = sum / spatial_size;
        }
    }

    Tensor4D::from_array(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
        // Avec ce kernel, copie le coin supérieur gauche de chaque patch
        assert_eq!(out_data[[0, 0, 0, 0]], 1.0);
        assert_eq!(out_data[[0, 0, 0, 1]], 2.0);
        assert_eq!(out_data[[0, 0, 1, 0]], 4.0);
        assert_eq!(out_data[[0, 0, 1, 1]], 5.0);
    }
}
