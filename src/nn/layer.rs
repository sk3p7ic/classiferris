use nalgebra::{DMatrix, DVector};

pub struct LayerShape {
    neurons: usize,
    m: usize,
}

pub struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    input_shape: LayerShape,
    output_shape: LayerShape,
}

impl Layer {
    pub fn new(neurons: usize, m: usize) -> Layer {
        let weights = DMatrix::zeros(neurons, m);
        let biases = DVector::zeros(neurons);
        let input_shape = LayerShape { neurons, m }; // TODO: Make proper
        let output_shape = LayerShape { neurons, m }; // TODO: Make proper
        Layer {
            weights,
            biases,
            input_shape,
            output_shape,
        }
    }
}
