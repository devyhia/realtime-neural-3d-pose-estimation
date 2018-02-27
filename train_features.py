import argparse
import random
import tensorflow as tf
from models.features import Features
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
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--log-path', type=str, default='/tmp/tensorboard', help='logging path for tensorboard')
parser.add_argument('--num-workers', type=int, default=2, help='how many workers for data loading')
parser.add_argument('--manual-seed', type=int, default=800, help='manual seed for random number generators')

use_gpu = is_gpu_available()

if __name__ == '__main__':
    args = parser.parse_args()

    tf.set_random_seed(args.manual_seed)
    random.seed(args.manual_seed)

    # Set up logger
    logger = setup_logger()

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

    # TF Optimizer / Adam Optimizer
    logger.info("Creating Adam Optimizer ...")
    optimizer = tf.train.AdamOptimizer(args.lr).minimize(model.graph['total_loss'])

    # Model Initialization
    init = tf.global_variables_initializer()

    # Start Training
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(init)

        # Tensorboard Initialization
        summary_writer = tf.summary.FileWriter(args.log_path, sess.graph)
        
        # Optimize :)
        for epoch in range(1, args.epochs + 1):
            batch_iterator = dataset.batch_training_triplets(args.batch_size)
            for batch_idx, data in enumerate(batch_iterator):
                anchors, pullers, pushers = \
                    data['anchor'], data['puller'], data['pusher']

                # Backpropagate gradients
                model.optimize(sess, optimizer, summary_writer, anchors, pullers, pushers)
                
                # Report on progress
                if batch_idx % args.log_interval == 0:
                    # Evaluate performance
                    loss = model.evaluate_triplet(anchors, pullers, pushers, session=sess)

                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * args.batch_size, dataset.training_length(),
                        (100. * batch_idx * args.batch_size) / dataset.training_length(), loss
                    ))
            
            # Save Model After Each Epoch
            save_path = model.save_model(sess, 'checkpoints/model.epoch.{}.ckpt'.format(epoch))
            logger.info("Model saved @ {}".format(save_path))