use crate::network::{Network, Activation, LossFunction, WeightInit};
use crate::optimizer::OptimizerType;
use crate::dataset::Dataset;
use crate::callbacks::{Callback, LearningRateScheduler};

/// Builder for constructing neural networks using a fluent interface.
///
/// # Example
/// ```rust
/// use cma_neural_network::builder::NetworkBuilder;
/// use cma_neural_network::network::{Activation, LossFunction};
/// use cma_neural_network::optimizer::OptimizerType;
///
/// let network = NetworkBuilder::new(2, 1)
///     .hidden_layer(8, Activation::ReLU)
///     .hidden_layer(5, Activation::ReLU)
///     .optimizer(OptimizerType::adam(0.001))
///     .loss(LossFunction::BinaryCrossEntropy)
///     .dropout(0.3)
///     .l2(0.01)
///     .build();
/// ```
pub struct NetworkBuilder {
    input_size: usize,
    output_size: usize,
    hidden_layers: Vec<(usize, Activation)>,
    output_activation: Activation,
    loss_function: LossFunction,
    optimizer: OptimizerType,
    weight_init: Option<WeightInit>,
    dropout_rate: Option<f64>,
    l1_lambda: Option<f64>,
    l2_lambda: Option<f64>,
    elastic_net: Option<(f64, f64)>, // (l1_ratio, lambda)
}

impl NetworkBuilder {
    /// Creates a new builder with the specified input and output sizes.
    ///
    /// Defaults:
    /// - Output activation: Sigmoid
    /// - Loss: BinaryCrossEntropy
    /// - Optimizer: Adam(0.001)
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            hidden_layers: Vec::new(),
            output_activation: Activation::Sigmoid,
            loss_function: LossFunction::BinaryCrossEntropy,
            optimizer: OptimizerType::adam(0.001),
            weight_init: None,
            dropout_rate: None,
            l1_lambda: None,
            l2_lambda: None,
            elastic_net: None,
        }
    }

    /// Adds a hidden layer with the specified size and activation.
    pub fn hidden_layer(mut self, size: usize, activation: Activation) -> Self {
        self.hidden_layers.push((size, activation));
        self
    }

    /// Configures the output layer activation function.
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    /// Configures the loss function.
    pub fn loss(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = loss_function;
        self
    }

    /// Configures the optimizer.
    pub fn optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Configures weight initialization (optional, auto-selected based on activation if not set).
    pub fn weight_init(mut self, init: WeightInit) -> Self {
        self.weight_init = Some(init);
        self
    }

    /// Configures dropout for all hidden layers.
    pub fn dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = Some(rate);
        self
    }

    /// Configures L1 regularization.
    pub fn l1(mut self, lambda: f64) -> Self {
        self.l1_lambda = Some(lambda);
        self
    }

    /// Configures L2 regularization.
    pub fn l2(mut self, lambda: f64) -> Self {
        self.l2_lambda = Some(lambda);
        self
    }

    /// Configures Elastic Net regularization.
    pub fn elastic_net(mut self, l1_ratio: f64, lambda: f64) -> Self {
        self.elastic_net = Some((l1_ratio, lambda));
        self
    }

    /// Builds the network.
    pub fn build(self) -> Network {
        if self.hidden_layers.is_empty() {
            panic!("Network must have at least one hidden layer. Use .hidden_layer() to add layers.");
        }

        let hidden_sizes: Vec<usize> = self.hidden_layers.iter().map(|(size, _)| *size).collect();
        let hidden_activations: Vec<Activation> = self.hidden_layers.iter().map(|(_, act)| *act).collect();

        // Déterminer les initialisations
        let (hidden_inits, output_init) = if let Some(init) = self.weight_init {
            // Utiliser la même initialisation pour toutes les couches
            (vec![init; hidden_sizes.len()], init)
        } else {
            // Initialisation automatique basée sur l'activation
            let hidden_inits: Vec<WeightInit> = hidden_activations.iter()
                .map(|&act| WeightInit::for_activation(act))
                .collect();
            let output_init = WeightInit::for_activation(self.output_activation);
            (hidden_inits, output_init)
        };

        let mut network = Network::new_deep_with_init(
            self.input_size,
            hidden_sizes,
            self.output_size,
            hidden_activations,
            self.output_activation,
            self.loss_function,
            hidden_inits,
            output_init,
            self.optimizer,
        );

        // Appliquer dropout si configuré
        if let Some(rate) = self.dropout_rate {
            use crate::network::DropoutConfig;
            let num_layers = network.layers.len();
            for i in 0..num_layers - 1 {
                network.layers[i].dropout = Some(DropoutConfig::new(rate));
            }
        }

        // Appliquer régularisation (priorité: elastic_net > l2 > l1)
        use crate::network::RegularizationType;
        if let Some((l1_ratio, lambda)) = self.elastic_net {
            network.regularization = RegularizationType::elastic_net(l1_ratio, lambda);
        } else if let Some(lambda) = self.l2_lambda {
            network.regularization = RegularizationType::l2(lambda);
        } else if let Some(lambda) = self.l1_lambda {
            network.regularization = RegularizationType::l1(lambda);
        }

        network
    }
}

/// Builder for training a neural network.
///
/// # Example
/// ```rust,ignore
/// use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
/// use cma_neural_network::network::Activation;
/// use cma_neural_network::callbacks::{EarlyStopping, ProgressBar};
/// use cma_neural_network::dataset::Dataset;
///
/// let mut network = NetworkBuilder::new(2, 1)
///     .hidden_layer(8, Activation::Tanh)
///     .build();
///
/// let history = network.trainer()
///     .train_data(&train_dataset)
///     .validation_data(&val_dataset)
///     .epochs(100)
///     .batch_size(32)
///     .callback(Box::new(EarlyStopping::new(10, 0.0001)))
///     .fit();
/// ```
pub struct TrainingBuilder<'a> {
    network: &'a mut Network,
    train_data: Option<&'a Dataset>,
    val_data: Option<&'a Dataset>,
    epochs: usize,
    batch_size: usize,
    callbacks: Vec<Box<dyn Callback>>,
    scheduler: Option<LearningRateScheduler>,
}

impl<'a> TrainingBuilder<'a> {
    /// Creates a new training builder.
    ///
    /// Defaults:
    /// - epochs: 100
    /// - batch_size: 32
    pub fn new(network: &'a mut Network) -> Self {
        Self {
            network,
            train_data: None,
            val_data: None,
            epochs: 100,
            batch_size: 32,
            callbacks: Vec::new(),
            scheduler: None,
        }
    }

    /// Configures the training data.
    pub fn train_data(mut self, dataset: &'a Dataset) -> Self {
        self.train_data = Some(dataset);
        self
    }

    /// Configures the validation data (optional).
    pub fn validation_data(mut self, dataset: &'a Dataset) -> Self {
        self.val_data = Some(dataset);
        self
    }

    /// Configures the number of epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Configures the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Adds a callback.
    pub fn callback(mut self, callback: Box<dyn Callback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Configures a learning rate scheduler.
    pub fn scheduler(mut self, scheduler: LearningRateScheduler) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Starts training.
    ///
    /// # Panics
    /// Panics if train_data has not been configured.
    pub fn fit(mut self) -> Vec<(f64, Option<f64>)> {
        let train_dataset = self.train_data.expect("train_data must be set before calling fit()");

        // Unified fit() call with optional scheduler
        let scheduler_ref = self.scheduler.as_mut();
        self.network.fit(
            train_dataset,
            self.val_data,
            self.epochs,
            self.batch_size,
            scheduler_ref,
            &mut self.callbacks,
        )
    }
}

/// Extension trait to add the `trainer()` method to Network.
pub trait NetworkTrainer {
    /// Creates a training builder for this network.
    fn trainer(&mut self) -> TrainingBuilder<'_>;
}

impl NetworkTrainer for Network {
    fn trainer(&mut self) -> TrainingBuilder<'_> {
        TrainingBuilder::new(self)
    }
}
