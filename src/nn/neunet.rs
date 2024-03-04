use super::layer::Layer;

struct CNN {
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
}
