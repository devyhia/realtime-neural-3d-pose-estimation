import json
import numpy as np
from . import ObjectsDataset
import random
from collections import namedtuple
from PIL import Image


class TrainingDataset(ObjectsDataset):
    Triplet = namedtuple('Triplet', ['anchor', 'puller', 'pusher'])
    TripletItem = namedtuple('TripletItem', ['klass', 'path', 'pose', 'image'])

    def __len__(self):
        return sum([ len(self.dataset_train[c]) for c in self.classes ])
    
    def __getitem__(self, pos):
        """Loads one training item (anchor, puller and pusher).
        
        Arguments:
            pos {tuple} -- (class, index)
        
        Returns:
            TrainingItem -- (anchor, puller, pusher)
        """

        (c, idx) = pos
        
        return self.get_triplets(c, idx)
    

    def make_triplet(self, c, tpl):
        return TrainingDataset.TripletItem(
            path=   tpl[0],
            pose=   tpl[1],
            klass=  c,
            image=  Image.open(tpl[0])
        )
    
    def get_triplets(self, c, idx):
        """Get triplets for training the CNN (i.e. to learn the feature space)

        Arguments:
            c {string} -- class
            idx {int} -- index of the training item

        Returns:
            Triplet -- a triplet of anchor, puller and pusher
        """

        # Anchor
        anchor = self.make_triplet(c, self.dataset_train[c][idx])

        # Puller
        anchor_to_coarse_distances = [
            item[1].distance(anchor.pose)
            for item in self.dataset_coarse[c]
        ]
        puller_idx = np.argmin(anchor_to_coarse_distances)
        puller = self.make_triplet(c, self.dataset_coarse[c][puller_idx])

        # Pusher
        pusher_from_the_same_class = random.randint(0, 1)

        if pusher_from_the_same_class == 1:
            pusher_idx = np.argmax(anchor_to_coarse_distances)
            pusher = self.make_triplet(c, self.dataset_coarse[c][pusher_idx])
        elif pusher_from_the_same_class == 0:
            class_options = list(set(self.classes) - {c})
            different_class = random.choice(class_options)
            pusher = self.make_triplet(
                different_class,
                random.choice(self.dataset_coarse[different_class])
            )

        return TrainingDataset.Triplet(anchor, puller, pusher)