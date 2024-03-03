mod utils {
    pub mod dataloader;
    pub mod msgprinter;
}
use std::time::Instant;
use utils::msgprinter::str_print;

fn main() {
    let data_prep_timer = Instant::now();
    str_print("Loading './data/mnist_train.csv'...", data_prep_timer);
    let mut train_dataset = utils::dataloader::Dataset::from_file(
        "./data/mnist_train.csv",
        utils::dataloader::DatasetType::Train,
    )
    .expect("file to exist and be parsable");
    str_print("Loaded './data/mnist_train.csv'...", data_prep_timer);
    str_print("Loading './data/mnist_test.csv'...", data_prep_timer);
    let mut test_dataset = utils::dataloader::Dataset::from_file(
        "./data/mnist_test.csv",
        utils::dataloader::DatasetType::Test,
    )
    .expect("file to exist and be parsable");
    str_print("Loaded './data/mnist_test.csv'...", data_prep_timer);
    str_print("Normalizing datasets...", data_prep_timer);
    train_dataset.normalize();
    test_dataset.normalize();
    str_print("Done.", data_prep_timer);
}
