//! # Tensor4D: Tenseur 4D pour les opérations CNN
//!
//! Structure de données pour images en batch: `[batch, channels, height, width]`
//!
//! ## Convention NCHW (PyTorch-style)
//!
//! - **N**: Batch size (nombre d'images)
//! - **C**: Channels (1 pour grayscale, 3 pour RGB)
//! - **H**: Height (hauteur en pixels)
//! - **W**: Width (largeur en pixels)

use ndarray::{Array2, Array4, s};
use serde::{Deserialize, Serialize};

/// Shape d'un tenseur 4D
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub fn new(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            channels,
            height,
            width,
        }
    }

    /// Nombre total d'éléments
    pub fn size(&self) -> usize {
        self.batch * self.channels * self.height * self.width
    }

    /// Shape pour une seule image (sans batch)
    pub fn image_size(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Après convolution avec kernel_size, stride, padding
    pub fn after_conv(
        &self,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let out_h = (self.height + 2 * padding - kernel_size) / stride + 1;
        let out_w = (self.width + 2 * padding - kernel_size) / stride + 1;
        Self::new(self.batch, out_channels, out_h, out_w)
    }

    /// Après pooling
    pub fn after_pool(&self, pool_size: usize, stride: usize) -> Self {
        let out_h = (self.height - pool_size) / stride + 1;
        let out_w = (self.width - pool_size) / stride + 1;
        Self::new(self.batch, self.channels, out_h, out_w)
    }

    /// Après global average pooling → [batch, channels, 1, 1]
    pub fn after_global_pool(&self) -> Self {
        Self::new(self.batch, self.channels, 1, 1)
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.batch, self.channels, self.height, self.width
        )
    }
}

/// Tenseur 4D avec opérations CNN
///
/// Layout mémoire: [batch, channels, height, width] (NCHW)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor4D {
    data: Array4<f64>,
}

impl Tensor4D {
    /// Crée un tenseur à partir d'un Array4
    pub fn from_array(data: Array4<f64>) -> Self {
        Self { data }
    }

    /// Crée un tenseur rempli de zéros
    pub fn zeros(shape: TensorShape) -> Self {
        Self {
            data: Array4::zeros((shape.batch, shape.channels, shape.height, shape.width)),
        }
    }

    /// Crée un tenseur rempli de uns
    pub fn ones(shape: TensorShape) -> Self {
        Self {
            data: Array4::ones((shape.batch, shape.channels, shape.height, shape.width)),
        }
    }

    /// Crée un tenseur avec des valeurs aléatoires uniformes dans [-1, 1]
    pub fn random(shape: TensorShape) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..shape.size())
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect();
        Self {
            data: Array4::from_shape_vec(
                (shape.batch, shape.channels, shape.height, shape.width),
                data,
            )
            .unwrap(),
        }
    }

    /// Shape du tenseur
    pub fn shape(&self) -> TensorShape {
        let dim = self.data.dim();
        TensorShape::new(dim.0, dim.1, dim.2, dim.3)
    }

    /// Accès aux données brutes
    pub fn data(&self) -> &Array4<f64> {
        &self.data
    }

    /// Accès mutable aux données
    pub fn data_mut(&mut self) -> &mut Array4<f64> {
        &mut self.data
    }

    /// Convertit en Array4
    pub fn into_array(self) -> Array4<f64> {
        self.data
    }

    /// Flatten: [batch, channels, height, width] → [batch, channels * height * width]
    pub fn flatten(&self) -> Array2<f64> {
        let shape = self.shape();
        let flat_size = shape.channels * shape.height * shape.width;

        // Reshape chaque image du batch en vecteur - optimisé sans allocation intermédiaire
        let mut result = Array2::zeros((shape.batch, flat_size));
        for b in 0..shape.batch {
            let image = self.data.slice(s![b, .., .., ..]);
            // Itère directement sans Vec intermédiaire
            for (i, &val) in image.iter().enumerate() {
                result[[b, i]] = val;
            }
        }
        result
    }

    /// Unflatten: [batch, flat] → [batch, channels, height, width]
    pub fn unflatten(flat: &Array2<f64>, shape: TensorShape) -> Self {
        let mut data = Array4::zeros((shape.batch, shape.channels, shape.height, shape.width));

        for b in 0..shape.batch {
            let mut idx = 0;
            for c in 0..shape.channels {
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        data[[b, c, h, w]] = flat[[b, idx]];
                        idx += 1;
                    }
                }
            }
        }

        Self { data }
    }

    /// Applique une fonction élément par élément
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Self {
            data: self.data.mapv(|x| f(x)),
        }
    }

    /// Applique ReLU
    pub fn relu(&self) -> Self {
        self.map(|x| x.max(0.0))
    }

    /// Somme sur tous les éléments
    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    /// Moyenne sur tous les éléments
    pub fn mean(&self) -> f64 {
        self.sum() / (self.shape().size() as f64)
    }

    /// Extrait une image du batch
    pub fn get_image(&self, batch_idx: usize) -> Array4<f64> {
        let image = self.data.slice(s![batch_idx..batch_idx + 1, .., .., ..]);
        image.to_owned()
    }

    /// Padding: ajoute des zéros autour de l'image
    pub fn pad(&self, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let shape = self.shape();
        let new_h = shape.height + 2 * padding;
        let new_w = shape.width + 2 * padding;

        let mut padded = Array4::zeros((shape.batch, shape.channels, new_h, new_w));

        // Copie les données au centre
        for b in 0..shape.batch {
            for c in 0..shape.channels {
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        padded[[b, c, h + padding, w + padding]] = self.data[[b, c, h, w]];
                    }
                }
            }
        }

        Self { data: padded }
    }
}

impl std::ops::Add for &Tensor4D {
    type Output = Tensor4D;

    fn add(self, other: &Tensor4D) -> Tensor4D {
        Tensor4D {
            data: &self.data + &other.data,
        }
    }
}

impl std::ops::Sub for &Tensor4D {
    type Output = Tensor4D;

    fn sub(self, other: &Tensor4D) -> Tensor4D {
        Tensor4D {
            data: &self.data - &other.data,
        }
    }
}

impl std::ops::Mul<f64> for &Tensor4D {
    type Output = Tensor4D;

    fn mul(self, scalar: f64) -> Tensor4D {
        Tensor4D {
            data: &self.data * scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(32, 3, 224, 224);
        assert_eq!(shape.batch, 32);
        assert_eq!(shape.channels, 3);
        assert_eq!(shape.height, 224);
        assert_eq!(shape.width, 224);
        assert_eq!(shape.size(), 32 * 3 * 224 * 224);
    }

    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new(2, 1, 28, 28);
        let tensor = Tensor4D::zeros(shape);
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.sum(), 0.0);
    }

    #[test]
    fn test_flatten_unflatten() {
        let shape = TensorShape::new(2, 3, 4, 4);
        let tensor = Tensor4D::ones(shape);

        let flat = tensor.flatten();
        assert_eq!(flat.dim(), (2, 48)); // 3 * 4 * 4 = 48

        let unflat = Tensor4D::unflatten(&flat, shape);
        assert_eq!(unflat.shape(), shape);
    }

    #[test]
    fn test_shape_after_conv() {
        let shape = TensorShape::new(1, 1, 28, 28);
        // Conv 5x5, stride 1, padding 0 → 28 - 5 + 1 = 24
        let after = shape.after_conv(32, 5, 1, 0);
        assert_eq!(after.height, 24);
        assert_eq!(after.width, 24);
        assert_eq!(after.channels, 32);
    }

    #[test]
    fn test_shape_after_pool() {
        let shape = TensorShape::new(1, 32, 24, 24);
        // Pool 2x2, stride 2 → 24 / 2 = 12
        let after = shape.after_pool(2, 2);
        assert_eq!(after.height, 12);
        assert_eq!(after.width, 12);
        assert_eq!(after.channels, 32);
    }

    #[test]
    fn test_padding() {
        let shape = TensorShape::new(1, 1, 3, 3);
        let tensor = Tensor4D::ones(shape);

        let padded = tensor.pad(1);
        assert_eq!(padded.shape().height, 5);
        assert_eq!(padded.shape().width, 5);
        // Coins doivent être 0
        assert_eq!(padded.data()[[0, 0, 0, 0]], 0.0);
        // Centre doit être 1
        assert_eq!(padded.data()[[0, 0, 1, 1]], 1.0);
    }
}
