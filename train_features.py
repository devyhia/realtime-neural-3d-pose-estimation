import argparse
import random
import tensorflow as tf
import numpy as np
from models.features import Features
from models.classifier import NearestNeighbour
from dataset import ObjectsDataset
from helpers.logger import setup_logger

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def is_gpu_available():
    return len(get_available_gpus()) > 0

# Training settings
parser = argparse.ArgumentParser(description='Feature Extractor Trainer')
parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
parser.add_argument('--dataset', default='/Users/yehyaa/Downloads/dataset/', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.95, help='momentum at which lr decreases')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--log-path', type=str, default='/tmp/tensorboard', help='logging path for tensorboard')
parser.add_argument('--num-workers', type=int, default=2, help='how many workers for data loading')
parser.add_argument('--manual-seed', type=int, default=800, help='manual seed for random number generators')

use_gpu = is_gpu_available()

# Set up logger
logger = setup_logger()

def dataset_in_feature_space(session, model, dataset, dataset_list, batch_size=16):
    dataset_features = []
    for batch in dataset.batch_items(dataset_list, batch_size, shuffle=False):
        dataset_features.append(model(session, batch))
    
    return np.concatenate(dataset_features)


def evaluate_model(session, model, dataset, batch_size=16):
    logger.info("Transforming features for coarse dataset ...")
    coarse_features = dataset_in_feature_space(session, model, dataset, dataset.dataset_coarse_list, batch_size)
    logger.info("Got {} features from coarse dataset(n= {}) ...".format(coarse_features.shape, len(dataset.dataset_coarse_list)))
    
    logger.info("Transforming features for test dataset ...")
    test_features = dataset_in_feature_space(session, model, dataset, dataset.dataset_test_list, batch_size)
    logger.info("Got {} features from coarse dataset(n= {}) ...".format(test_features.shape, len(dataset.dataset_test_list)))

    logger.info("Creating classifier using coarse features ...")
    classifier = NearestNeighbour(coarse_features, dataset.dataset_coarse_list)

    histogram = {
        10: 0,
        20: 0,
        40: 0,
        180: 0
    }

    for idx in range(test_features.shape[0]):
        test_label, (_, test_quanternion) = dataset.dataset_test_list[idx]
        feature_vector = test_features[idx, :]
        prediction_label, (_, prediction_quaternion) = classifier.match(feature_vector)

        if test_label != prediction_label:
            continue
        
        # Both labels match-- Build the histogram
        angle = test_quanternion.distance(prediction_quaternion)
        if angle <= np.pi / 18:
            histogram[10] += 1
        
        if angle <= 2 * np.pi / 18:
            histogram[2 * 10] += 1
        
        if angle <= 4 * np.pi / 18:
            histogram[4 * 10] += 1
        
        if angle <= 18 * np.pi / 18:
            histogram[18 * 10] += 1
    
    return histogram


if __name__ == '__main__':
    args = parser.parse_args()

    tf.set_random_seed(args.manual_seed)
    random.seed(args.manual_seed)

    logger.info(args)

    # Load Dataset & Batch Loader
    logger.info("Loading the dataset ...")
    dataset = ObjectsDataset(args.dataset)

    # Load Model
    logger.info("Loading the model ...")
    model = Features()

    # Enable GPU
    device = '/gpu:0' if use_gpu else '/cpu:0'
    logger.info("Training using {} ...".format(device))

    # Model Initialization
    init = tf.global_variables_initializer()

    # Start Training
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(init)

        # Tensorboard Initialization
        summary_writer = tf.summary.FileWriter(args.log_path, sess.graph)
        
        learning_rate = args.lr
        
        # Optimize :)
        for epoch in range(1, args.epochs + 1):
            batch_iterator = dataset.batch_training_triplets(args.batch_size)
            for batch_idx, data in enumerate(batch_iterator):
                anchors, pullers, pushers = \
                    data['anchor'], data['puller'], data['pusher']

                # Backpropagate gradients
                model.optimize(sess, summary_writer, learning_rate, anchors, pullers, pushers)
                
                # Report on progress
                if batch_idx % args.log_interval == 0:
                    # Evaluate performance
                    loss = model.evaluate_triplet(anchors, pullers, pushers, session=sess)

                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * args.batch_size, dataset.training_length(),
                        (100. * batch_idx * args.batch_size) / dataset.training_length(), loss
                    ))
                
                if (batch_idx + 1) % (10 * args.log_interval) == 0:
                    # Evaluate the Model
                    histogram = evaluate_model(sess, model, dataset, args.batch_size)
                    logger.info("Model histgram is: {}".format(histogram))
            
            # Evaluate the Model
            histogram = evaluate_model(sess, model, dataset, args.batch_size)
            logger.info("Model histgram is: {}".format(histogram))

            # Save Model After Each Epoch
            save_path = model.save_model(sess, 'checkpoints/model.epoch.{}.ckpt'.format(epoch))
            logger.info("Model saved @ {}".format(save_path))

            # Adaptive Learning Rate
            learning_rate *= args.momentum
            logger.info("New Learning Rate: {}".format(learning_rate))