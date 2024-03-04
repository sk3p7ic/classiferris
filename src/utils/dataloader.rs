use nalgebra::base::{DMatrix, DVector};
use std::convert::From;

#[derive(Clone, Copy, Debug)]
pub enum DatasetType {
    Train,
    Test,
}

impl DatasetType {
    pub fn get_n_items(&self) -> usize {
        match self {
            Self::Test => 10000,
            Self::Train => 60000,
        }
    }
}

pub struct Dataset {
    pub t: DatasetType,
    pub labels: Vec<DVector<u8>>,
    pub images: Vec<DMatrix<f32>>,
    pub n_items: usize,
}

impl Dataset {
    pub fn from_file(file_path: &str, t: DatasetType, batch_size: usize) -> Result<Dataset, String> {
        if t.get_n_items() % batch_size != 0 {
            return Err(format!("Dataset size '{}' not divisible by batch size '{}'.", t.get_n_items(), batch_size).to_string());
        }
        let file_contents = std::fs::read_to_string(file_path)
            .expect("file to exist. Please download dataset files.");
        let labels_vec = file_contents
            .trim()
            .split('\n')
            .map(|ln| {
                ln.trim()
                    .split(',')
                    .nth(0)
                    .expect("line to have label value")
                    .trim()
                    .parse::<u8>()
                    .expect("value to be valid u8.")
            })
            .collect::<Vec<u8>>()
            .chunks(batch_size)
            .map(|c| c.to_owned())
            .collect::<Vec<Vec<u8>>>();
        let images_vec = file_contents
            .trim()
            .split('\n')
            .map(|ln| {
                ln.trim()
                    .split(',')
                    .skip(1)
                    .map(|c| f32::from(c.trim().parse::<u8>().expect("value to be valid u8.")))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>()
            .chunks(batch_size)
            .map(|c| c.to_owned())
            .collect::<Vec<Vec<Vec<f32>>>>();
        let labels = labels_vec.into_iter()
            .map(|c| DVector::<u8>::from_vec(c))
            .collect();
        let images = images_vec.into_iter()
            .map(|c| DMatrix::<f32>::from_vec(
                batch_size,
                784,
                c.into_iter().flatten().collect()
            ))
            .collect();
        Ok(Dataset {
            t,
            labels,
            images,
            n_items: t.get_n_items(),
        })
    }

    pub fn normalize(&mut self) {
        self.images.iter_mut().for_each(|img_v| img_v.apply(|px| *px /= 255f32));
    }
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "Dataset ({:?}) {{
            labels: {}x{:?},
            images: {}x{:?},
            n_items: {}
        }}",
                self.t,
                self.labels.len(),
                self.labels[0].shape(),
                self.images.len(),
                self.images[0].shape(),
                self.n_items
            )
            .as_str(),
        )
    }
}
