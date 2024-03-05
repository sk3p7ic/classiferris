mod nn {
    pub mod layer;
    pub mod neunet;
}
mod utils {
    pub mod dataloader;
    pub mod msgprinter;
}
use nn::{layer::{Layer, LayerType, ActivationFunction}, neunet::CNN};
use utils::msgprinter::str_print;
use std::time::Instant;

fn main() {
    let data_prep_timer = Instant::now();
    str_print("Loading './data/mnist_train.csv'...", data_prep_timer);
    let mut train_dataset = utils::dataloader::Dataset::from_file(
        "./data/mnist_train.csv",
        utils::dataloader::DatasetType::Train,
        40
    )
    .expect("file to exist and be parsable");
    str_print("Loaded './data/mnist_train.csv'...", data_prep_timer);
    str_print(format!("{}", train_dataset).as_str(), data_prep_timer);
    str_print("Loading './data/mnist_test.csv'...", data_prep_timer);
    let mut test_dataset = utils::dataloader::Dataset::from_file(
        "./data/mnist_test.csv",
        utils::dataloader::DatasetType::Test,
        40
    )
    .expect("file to exist and be parsable");
    str_print("Loaded './data/mnist_test.csv'...", data_prep_timer);
    str_print(format!("{}", test_dataset).as_str(), data_prep_timer);
    str_print("Normalizing datasets...", data_prep_timer);
    train_dataset.normalize();
    test_dataset.normalize();
    str_print("Done.", data_prep_timer);
    str_print("Building neural network...", data_prep_timer);
    let mut nn = CNN::init();
    nn.add_layer(Layer::new(784, 392, LayerType::Input, ActivationFunction::None));
    nn.add_layer(Layer::new(392, 98, LayerType::Hidden, ActivationFunction::ReLU));
    nn.add_layer(Layer::new(98, 10, LayerType::Hidden, ActivationFunction::ReLU));
    nn.add_layer(Layer::new(10, 1, LayerType::Output, ActivationFunction::Softmax));
    str_print("Added layers", data_prep_timer);
    str_print(format!("{}", nn).as_str(), data_prep_timer);
}
