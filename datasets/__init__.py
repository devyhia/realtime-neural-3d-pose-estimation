from torch.utils.data import Dataset
from glob import glob
from helpers.quaternion import Quaternion
from collections import namedtuple
import numpy as np


class ObjectsDataset(Dataset):
    """A PyTorch wrapper for the Objects dataset
    """

    classes = ['ape', 'benchvise', 'cam', 'cat', 'duck']

    def __init__(self, dataset_dir):
        """Initializes the ObjectsDataset: Loads the images and their 
        respective poses (i.e. Quaternions)

        Arguments:
            dataset_dir {string} -- location of the dataset folder
        """
        
        self.mean = np.array([63.96652548, 54.81466454, 48.04923144])[:, np.newaxis, np.newaxis]

        with open('{}real/training_split.txt'.format(dataset_dir)) as f:
            training_split = f.readline()
            training_indices = {
                int(idx.strip())
                for idx in training_split.strip().split(',')
            }

        dataset_train = {}
        dataset_test = {}
        dataset_coarse = {}

        for c in ObjectsDataset.classes:
            dataset_train[c] = []
            dataset_test[c] = []
            dataset_coarse[c] = []

            images = glob('/{}/real/{}/*.png'.format(dataset_dir, c))
            all_indices = set(range(len(images)))

            with open('{}real/{}/poses.txt'.format(dataset_dir, c)) as f:
                poses = f.readlines()

            # Training set from "real"
            for idx in training_indices:
                pose = Quaternion(
                    *map(float, poses[2 * idx + 1].strip().split(' ')))
                image = '{}real/{}/real{}.png'.format(dataset_dir, c, idx)

                dataset_train[c].append((image, pose))

            # Testing set from "real"
            for idx in all_indices - training_indices:
                if 2 * idx + 1 > len(poses) - 1:
                    # If the image has no pose, then skip it!
                    continue

                pose = Quaternion(
                    *map(float, poses[2 * idx + 1].strip().split(' ')))
                image = '{}real/{}/real{}.png'.format(dataset_dir, c, idx)

                dataset_test[c].append((image, pose))

            # Training set from "fine"
            images = glob('{}fine/{}/*.png'.format(dataset_dir, c))
            with open('{}fine/{}/poses.txt'.format(dataset_dir, c)) as f:
                poses = f.readlines()

            for idx in range(len(images)):
                pose = Quaternion(
                    *map(float, poses[2 * idx + 1].strip().split(' ')))
                image = '{}fine/{}/fine{}.png'.format(dataset_dir, c, idx)
                dataset_train[c].append((image, pose))

            # Database set from "coarse"
            images = glob('{}coarse/{}/*.png'.format(dataset_dir, c))
            with open('{}coarse/{}/poses.txt'.format(dataset_dir, c)) as f:
                poses = f.readlines()

            for idx in range(len(images)):
                pose = Quaternion(
                    *map(float, poses[2 * idx + 1].strip().split(' ')))
                image = '{}coarse/{}/coarse{}.png'.format(dataset_dir, c, idx)
                dataset_coarse[c].append((image, pose))

        self.dataset_test = dataset_test
        self.dataset_coarse = dataset_coarse
        self.dataset_train = dataset_train
        self.classes = ObjectsDataset.classes
