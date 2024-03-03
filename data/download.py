from dataclasses import dataclass
import numpy as np
import requests
import gzip
import os


@dataclass
class Download:
    url: str
    filename: str

    def _get(self) -> None:
        print(f"Downloading {self.url} to {self.filename}...")
        response = requests.get(self.url)
        with open(self.filename, "wb") as file:
            file.write(response.content)
        print(f"Done.")

    def gunzip(self) -> bytes:
        if not os.path.exists(self.filename):
            self._get()
        else:
            print(f"{self.filename} already exists. Skipping download.")
        with gzip.open(self.filename, "rb") as file:
            data = file.read()
        return data


@dataclass
class DatasetDownload:
    url: str
    train_images: str
    train_labels: str
    test_images: str
    test_labels: str

    def download(self) -> dict[str, bytes]:
        return {
            "train_images": Download(
                self.url + self.train_images, self.train_images
            ).gunzip(),
            "train_labels": Download(
                self.url + self.train_labels, self.train_labels
            ).gunzip(),
            "test_images": Download(
                self.url + self.test_images, self.test_images
            ).gunzip(),
            "test_labels": Download(
                self.url + self.test_labels, self.test_labels
            ).gunzip(),
        }


class Dataset:
    train_images: np.ndarray | None = None
    train_labels: np.ndarray | None = None
    test_images: np.ndarray | None = None
    test_labels: np.ndarray | None = None

    def download_and_unpack(self, dl_source: DatasetDownload) -> None:
        print("Downloading and unpacking dataset...")
        data = dl_source.download()
        self.train_images = np.frombuffer(
            data["train_images"], dtype=np.uint8, offset=16
        ).reshape(-1, 28, 28)
        self.train_labels = np.frombuffer(
            data["train_labels"], dtype=np.uint8, offset=8
        )
        self.test_images = np.frombuffer(
            data["test_images"], dtype=np.uint8, offset=16
        ).reshape(-1, 28, 28)
        self.test_labels = np.frombuffer(data["test_labels"], dtype=np.uint8, offset=8)
        print(
            f"Done. Train images: {self.train_images.shape=}, test images: {self.test_images.shape=}"
        )

    def save(self, train_path: str, test_path: str) -> bool:
        if (
            type(self.train_images) == None
            or type(self.train_labels) == None
            or type(self.test_images) == None
            or type(self.test_labels) == None
        ):
            return False
        train_data = np.concatenate(
            [self.train_labels[:, None], self.train_images.reshape(-1, 28 * 28)], axis=1
        )
        test_data = np.concatenate(
            [self.test_labels[:, None], self.test_images.reshape(-1, 28 * 28)], axis=1
        )
        with open(train_path, "w") as train_file:
            print(f"Saving train data to {train_path}...")
            for row in train_data:
                train_file.write(",".join(map(str, row)) + "\n")
        with open(test_path, "w") as test_file:
            print(f"Saving test data to {test_path}...")
            for row in test_data:
                test_file.write(",".join(map(str, row)) + "\n")
        return True


if __name__ == "__main__":
    dl_source = DatasetDownload(
        url="http://yann.lecun.com/exdb/mnist/",
        train_images="train-images-idx3-ubyte.gz",
        train_labels="train-labels-idx1-ubyte.gz",
        test_images="t10k-images-idx3-ubyte.gz",
        test_labels="t10k-labels-idx1-ubyte.gz",
    )
    dataset = Dataset()
    dataset.download_and_unpack(dl_source)
    dataset.save("mnist_train.csv", "mnist_test.csv")
