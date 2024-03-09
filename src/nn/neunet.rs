use nalgebra::DMatrix;

use super::layer::Layer;

pub struct CNN {
    layers: Vec<Layer>,
}

impl CNN {
    pub fn init() -> CNN {
        // TODO: Make take input and output layers
        CNN { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, input: DMatrix<f32>) -> Result<u8, String> {
        if self.layers.len() < 2 {
            return Err("Not enough layers for forwarding.".to_string());
        }
        let mut logit = self.layers[0].forward(input);
        for i in 1..self.layers.len() {
            logit = self.layers[i].forward(logit);
        }
        Ok(self.predict(logit))
    }

    fn predict(&self, mtx: DMatrix<f32>) -> u8 {
        let mut max = 0f32;
        let mut max_idx = 0;
        for i in 0..mtx.nrows() {
            if &max < mtx.get((0, i)).unwrap() {
                max = mtx.get((0, i)).unwrap().clone();
                max_idx = i as u8;
            }
        }
        max_idx
    }
}

impl std::fmt::Display for CNN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let layer_strs = self.layers.iter()
            .map(|lyr| format!("\t\t{},\n", lyr))
            .collect::<String>();
        f.write_str(format!("CNN {{
            Layers: [\n{}\n\t]
        }}", layer_strs).as_str())
    }
}
