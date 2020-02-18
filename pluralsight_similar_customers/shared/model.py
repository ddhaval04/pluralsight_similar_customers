from abc import ABC, abstractmethod


class Model(ABC):
    # TODO: add model-name and model-version attributes?
    # mutable attributes
    @abstractmethod
    def is_trained(self):
        pass

    @abstractmethod
    def models_folder(self):
        pass

    @abstractmethod
    def models_filename(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load_models_from_files(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def predict_on_batch(self, data_batch):
        pass
