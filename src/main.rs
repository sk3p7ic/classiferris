mod utils {
    pub mod dataloader;
}
use std::time::Instant;

fn main() {
    println!("Loading training dataset...");
    let train_start_instant = Instant::now();
    let mut train_dataset = utils::dataloader::Dataset::from_file(
        "./data/mnist_train.csv",
        utils::dataloader::DatasetType::Train,
    )
    .expect("file to exist and be parsable");
    println!(
        "Loaded training dataset in {:.2}s.\nNormalizing training dataset...",
        train_start_instant.elapsed().as_secs_f32()
    );
    let train_norm_instant = Instant::now();
    train_dataset.normalize();
    println!(
        "Done. Normalization took {:.2}s.",
        train_norm_instant.elapsed().as_secs_f32()
    );
}
