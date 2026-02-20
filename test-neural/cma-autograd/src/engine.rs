//! # Backward Engine
//!
//! Orchestration du backward pass :
//! 1. Tri topologique du graphe de calcul
//! 2. Propagation des gradients depuis la loss vers les feuilles
//! 3. Accumulation des gradients dans les tenseurs feuilles

use crate::Float;
use crate::grad_fn::GradFn;
use crate::tensor::{Tensor, set_grad_enabled};
use ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Execute backward pass starting from the given (scalar) tensor.
///
/// This is the core engine that:
/// 1. Initializes the gradient of the root tensor to 1.0
/// 2. Performs a topological sort of the computation graph
/// 3. Propagates gradients backward through each operation
/// 4. Accumulates gradients in leaf tensors
pub fn backward(root: &Tensor) {
    assert!(
        root.numel() == 1,
        "backward() can only be called on a scalar tensor (got {} elements)",
        root.numel()
    );

    // Start with gradient of 1.0 for the root
    let initial_grad = ArrayD::from_elem(IxDyn(&[]), 1.0 as Float);

    // Build the topological ordering of the graph
    let topo_order = topological_sort(root);

    // Map from tensor id to accumulated gradient
    let mut grads: HashMap<usize, ArrayD<Float>> = HashMap::new();
    grads.insert(root.id(), initial_grad);

    // Traverse in reverse topological order (root first, leaves last)
    for (tensor, grad_fn) in &topo_order {
        // Get the accumulated gradient for this tensor
        let grad_output = match grads.get(&tensor.id()) {
            Some(g) => g.clone(),
            None => continue,
        };

        // Compute gradients for inputs
        let input_grads = grad_fn.backward(&grad_output);
        let inputs = grad_fn.inputs();

        assert_eq!(
            input_grads.len(),
            inputs.len(),
            "GradFn {} returned {} gradients but has {} inputs",
            grad_fn.name(),
            input_grads.len(),
            inputs.len()
        );

        // Accumulate gradients into input tensors
        for (input, grad) in inputs.iter().zip(input_grads.iter()) {
            if !input.requires_grad() {
                continue;
            }

            let id = input.id();
            grads
                .entry(id)
                .and_modify(|existing| *existing += grad)
                .or_insert_with(|| grad.clone());

            // If it's a leaf tensor, also store in the tensor itself
            if input.is_leaf() {
                input.accumulate_grad(grad);
            }
        }
    }
}

/// Builds a topological ordering of the computation graph.
///
/// Returns pairs of (Tensor, GradFn) in order from root to leaves.
fn topological_sort(root: &Tensor) -> Vec<(Tensor, Arc<dyn GradFn>)> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn visit(
        tensor: &Tensor,
        visited: &mut HashSet<usize>,
        order: &mut Vec<(Tensor, Arc<dyn GradFn>)>,
    ) {
        let id = tensor.id();
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        if let Some(grad_fn) = tensor.grad_fn() {
            // Visit all inputs first (DFS post-order)
            for input in grad_fn.inputs() {
                visit(&input, visited, order);
            }
            order.push((tensor.clone(), grad_fn));
        }
    }

    visit(root, &mut visited, &mut order);

    // Reverse: root first for backward traversal
    order.reverse();
    order
}

// ═══════════════════════════════════════════════════════════════════════════
// no_grad context
// ═══════════════════════════════════════════════════════════════════════════

/// Execute a closure with gradient computation disabled.
///
/// Useful for inference or when updating parameters to avoid tracking.
///
/// # Example
///
/// ```rust,ignore
/// use cma_autograd::engine::no_grad;
///
/// let prediction = no_grad(|| {
///     model.forward(&input)
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = set_grad_enabled(false);
    let result = f();
    set_grad_enabled(prev);
    result
}

/// RAII guard that disables gradient computation while alive.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    /// Create a guard that disables gradient computation.
    pub fn new() -> Self {
        let prev = set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_no_grad() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2], true);
        let b = Tensor::from_vec(vec![3.0, 4.0], &[2], true);

        // With grad enabled
        let c = &a + &b;
        assert!(c.grad_fn().is_some());

        // With grad disabled
        let d = no_grad(|| &a + &b);
        assert!(d.grad_fn().is_none());
        assert!(!d.requires_grad());
    }

    #[test]
    fn test_no_grad_guard() {
        let a = Tensor::from_vec(vec![1.0], &[1], true);
        let b = Tensor::from_vec(vec![2.0], &[1], true);

        {
            let _guard = NoGradGuard::new();
            let c = &a + &b;
            assert!(!c.requires_grad());
        }

        // After guard is dropped, grad is re-enabled
        let d = &a + &b;
        assert!(d.grad_fn().is_some());
    }
}
