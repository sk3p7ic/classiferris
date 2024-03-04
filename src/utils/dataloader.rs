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
    pub labels: DVector<u8>,
    pub images: DMatrix<f32>,
    pub n_items: usize,
}

impl Dataset {
    pub fn from_file(file_path: &str, t: DatasetType) -> Result<Dataset, String> {
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
            .collect::<Vec<u8>>();
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
            .collect::<Vec<Vec<f32>>>();
        let labels = DVector::from_vec(labels_vec);
        let images = DMatrix::from_vec(
            t.get_n_items(),
            784,
            images_vec.into_iter().flatten().collect::<Vec<f32>>(),
        );
        Ok(Dataset {
            t,
            labels,
            images,
            n_items: t.get_n_items(),
        })
    }

    pub fn normalize(&mut self) {
        self.images.apply(|px| *px /= 255f32);
    }
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "Dataset ({:?}) {{
            labels: {:?},
            images: {:?},
            n_items: {}
        }}",
                self.t,
                self.labels.shape(),
                self.images.shape(),
                self.n_items
            )
            .as_str(),
        )
    }
}
