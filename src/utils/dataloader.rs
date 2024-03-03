use std::convert::From;

#[derive(Clone, Copy)]
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
    pub labels: Vec<u8>,
    pub images: Vec<Vec<f32>>,
    pub n_items: usize,
}

impl Dataset {
    pub fn from_file(file_path: &str, t: DatasetType) -> Result<Dataset, String> {
        let file_contents = std::fs::read_to_string(file_path)
            .expect("file to exist. Please download dataset files.");
        let labels = file_contents
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
        let images = file_contents
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
        Ok(Dataset {
            t,
            labels,
            images,
            n_items: t.get_n_items(),
        })
    }

    pub fn normalize(&mut self) {
        self.images = self
            .images
            .iter()
            .map(|img| img.iter().map(|&px| px / 255f32).collect())
            .collect();
    }
}
