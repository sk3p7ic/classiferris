use nalgebra::DMatrix;
use rand::Rng;

#[derive(Debug)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug)]
pub struct LayerShape {
    input: usize,
    output: usize,
}

pub struct Layer {
    activation: ActivationFunction,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    shape: LayerShape,
    t: LayerType,
}

impl Layer {
    pub fn new(input_neurons: usize, output_neurons: usize, t: LayerType, activation: ActivationFunction) -> Layer {
        let mut weights = DMatrix::zeros(output_neurons, input_neurons);
        let mut biases = DMatrix::zeros(1, output_neurons);
        let shape = LayerShape {
            input: input_neurons,
            output: output_neurons,
        };
        let mut rng = rand::thread_rng();
        weights.apply(|w| *w = rng.gen::<f32>() - 0.5);
        biases.apply(|b| *b = rng.gen::<f32>());
        Layer {
            activation,
            weights,
            biases,
            shape,
            t,
        }
    }

    pub fn forward(&self, input: DMatrix<f32>) -> DMatrix<f32> {
        (&self.weights * input) + &self.biases
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("Layer ({:?}) {{
            Weights: {:?},
            Biases: {:?},
            Shape: {:?}
        }}", self.t, self.weights.shape(), self.biases.shape(), self.shape)
            .as_str())
    }
}

#[derive(Debug)]
pub enum ActivationFunction {
    None,
    ReLU,
    Softmax
}

impl ActivationFunction {
    pub fn apply(&self, data: &mut DMatrix<f32>) {
        let sum: f32 = match self {
            Self::None => 0.0,
            Self::ReLU => 0.0,
            Self::Softmax => data.iter().sum()
        };
        data.apply(|v: &mut f32| {
            match self {
                Self::None => (),
                Self::ReLU => {if *v <= 0f32 { *v = 0f32; }},
                Self::Softmax => {
                    *v = v.exp() / sum;
                },
            }
        });
    }
}
