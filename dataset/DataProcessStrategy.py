from abc import ABC, abstractmethod


class DataProcessStrategy(ABC):
    @abstractmethod
    def prepare_data(self, train_data, test_data):
        pass

class WithTimeFeature(DataProcessStrategy):
    def prepare_data(self, train_data, test_data):
        pass

class WithoutTimeFeature(DataProcessStrategy):
    def prepare_data(self, train_data, test_data):
        pass