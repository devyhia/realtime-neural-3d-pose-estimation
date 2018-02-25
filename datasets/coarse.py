from . import ObjectsDataset

class CoarseDataset(ObjectsDataset):
    def __len__(self):
        return sum([ len(self.dataset_coarse[c]) for c in self.classes ])