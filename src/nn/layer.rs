use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub enum LayerType {
    Input,
    Hidden,
    Output,
}

pub struct LayerShape {
    input: usize,
    output: usize,
}

pub struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    shape: LayerShape,
    t: LayerType,
}

impl Layer {
    pub fn new(input_neurons: usize, output_neurons: usize, t: LayerType) -> Layer {
        let mut weights = DMatrix::zeros(output_neurons, input_neurons);
        let mut biases = DVector::zeros(output_neurons);
        let shape = LayerShape {
            input: input_neurons,
            output: output_neurons,
        };
        let mut rng = rand::thread_rng();
        weights.apply(|w| *w = rng.gen::<f32>() - 0.5);
        biases.apply(|b| *b = rng.gen::<f32>());
        Layer {
            weights,
            biases,
            shape,
            t,
        }
    }
}
