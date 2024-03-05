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
