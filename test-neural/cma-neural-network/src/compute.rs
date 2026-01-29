//! Compute device abstraction for training execution.
//!
//! This module provides a device-agnostic way to execute neural network training.
//! Currently supports CPU execution (single-threaded and parallel), with GPU support planned.
//!
//! # Architecture
//!
//! The `ComputeDevice` enum allows selecting the execution backend at runtime,
//! without polluting the core `Network` code with device-specific logic.
//!
//! # Example
//!
//! ```rust,ignore
//! use cma_neural_network::compute::ComputeDevice;
//!
//! // Default: CPU single-threaded execution
//! network.trainer()
//!     .device(ComputeDevice::Cpu)
//!     .train_data(&dataset)
//!     .fit();
//!
//! // Parallel CPU execution (requires "parallel" feature)
//! network.trainer()
//!     .device(ComputeDevice::CpuParallel)
//!     .train_data(&dataset)
//!     .fit();
//! ```

use std::fmt;

/// Compute device for training execution.
///
/// Determines where the training computations are performed.
/// This is a runtime choice that doesn't affect the network architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeDevice {
    /// CPU execution (default, single-threaded).
    ///
    /// Uses standard Rust/ndarray operations.
    /// Compatible with WebAssembly.
    #[default]
    Cpu,

    /// CPU execution with parallel processing (multi-threaded).
    ///
    /// Uses Rayon for parallel batch gradient computation.
    /// Requires the "parallel" feature to be enabled.
    /// NOT compatible with WebAssembly.
    CpuParallel,

    /// GPU execution (planned, not yet available).
    ///
    /// Will return an error when used, as GPU support is not implemented.
    Gpu,
}

impl ComputeDevice {
    /// Returns true if this device is available for use.
    pub fn is_available(&self) -> bool {
        match self {
            ComputeDevice::Cpu => true,
            #[cfg(feature = "parallel")]
            ComputeDevice::CpuParallel => true,
            #[cfg(not(feature = "parallel"))]
            ComputeDevice::CpuParallel => false,
            ComputeDevice::Gpu => false, // Not yet implemented
        }
    }

    /// Validates that this device can be used.
    /// Returns an error if the device is not available.
    pub fn validate(&self) -> Result<(), ComputeDeviceError> {
        match self {
            ComputeDevice::Cpu => Ok(()),
            #[cfg(feature = "parallel")]
            ComputeDevice::CpuParallel => Ok(()),
            #[cfg(not(feature = "parallel"))]
            ComputeDevice::CpuParallel => Err(ComputeDeviceError::ParallelNotEnabled),
            ComputeDevice::Gpu => Err(ComputeDeviceError::GpuNotAvailable),
        }
    }

    /// Returns the device name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            ComputeDevice::Cpu => "CPU",
            ComputeDevice::CpuParallel => "CPU (Parallel)",
            ComputeDevice::Gpu => "GPU",
        }
    }

    /// Returns true if this device uses parallel processing.
    pub fn is_parallel(&self) -> bool {
        matches!(self, ComputeDevice::CpuParallel)
    }
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Errors related to compute device operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeDeviceError {
    /// GPU execution is not available.
    ///
    /// GPU support is planned but not yet implemented.
    /// Use `ComputeDevice::Cpu` instead.
    GpuNotAvailable,

    /// Parallel CPU execution requires the "parallel" feature.
    ///
    /// Enable with: `cma-neural-network = { features = ["parallel"] }`
    ParallelNotEnabled,
}

impl fmt::Display for ComputeDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeDeviceError::GpuNotAvailable => {
                write!(
                    f,
                    "GPU compute is not available. GPU support is planned but not yet implemented. Use ComputeDevice::Cpu instead."
                )
            }
            ComputeDeviceError::ParallelNotEnabled => {
                write!(
                    f,
                    "Parallel CPU requires the 'parallel' feature. Enable with: cma-neural-network = {{ features = [\"parallel\"] }}"
                )
            }
        }
    }
}

impl std::error::Error for ComputeDeviceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_is_available() {
        assert!(ComputeDevice::Cpu.is_available());
        assert!(ComputeDevice::Cpu.validate().is_ok());
    }

    #[test]
    fn test_gpu_not_available() {
        assert!(!ComputeDevice::Gpu.is_available());
        assert!(ComputeDevice::Gpu.validate().is_err());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_available_with_feature() {
        assert!(ComputeDevice::CpuParallel.is_available());
        assert!(ComputeDevice::CpuParallel.validate().is_ok());
    }

    #[test]
    #[cfg(not(feature = "parallel"))]
    fn test_parallel_not_available_without_feature() {
        assert!(!ComputeDevice::CpuParallel.is_available());
        assert!(ComputeDevice::CpuParallel.validate().is_err());
    }

    #[test]
    fn test_default_is_cpu() {
        assert_eq!(ComputeDevice::default(), ComputeDevice::Cpu);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ComputeDevice::Cpu), "CPU");
        assert_eq!(format!("{}", ComputeDevice::Gpu), "GPU");
    }
}
