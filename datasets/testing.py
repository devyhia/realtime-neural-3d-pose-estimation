from . import ObjectsDataset

class TestingDataset(ObjectsDataset):
    def __len__(self):
        return sum([ len(self.dataset_test[c]) for c in self.classes ])