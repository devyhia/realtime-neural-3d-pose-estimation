import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

from models.features import Features
from models.classifier import NearestNeighbour
from dataset import ObjectsDataset
from helpers.logger import setup_logger
from helpers import dataset_in_feature_space

def save_metadata(file, database_list):
    with open(file, 'w') as f:
        for c, _ in database_list:
            f.write('{}\n'.format(c))

# Set up logger
logger = setup_logger()

# load model
logger.info("Creating Session ...")
sess = tf.InteractiveSession()

logger.info("Building Graph ...")
model = Features()

logger.info("Loading Checkpoint ...")
model.load_model(sess, 'checkpoints_gpu/model.epoch.50.ckpt')

logger.info("Loading Dataset ...")
dataset = ObjectsDataset('/Users/yehyaa/Downloads/dataset/')

logger.info("Coarse Features ...")
coarse_features = dataset_in_feature_space(sess, model, dataset, dataset.dataset_coarse_list, 4)

logger.info("Test Features ...")
test_features = dataset_in_feature_space(sess, model, dataset, dataset.dataset_test_list, 4)

logger.info("Constructing Embedding Model ...")
# setup a TensorFlow session
X_coarse = tf.Variable([0.0], name='embedding_coarse')
X_test = tf.Variable([0.0], name='embedding_test')
place_coarse = tf.placeholder(tf.float32, shape=coarse_features.shape)
place_test = tf.placeholder(tf.float32, shape=test_features.shape)
set_x_coarse = tf.assign(X_coarse, place_coarse, validate_shape=False)
set_x_test = tf.assign(X_test, place_test, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run([
    set_x_coarse,
    set_x_test
], feed_dict={
    place_coarse: coarse_features,
    place_test: test_features
})

# write labels
logger.info("Writing coarse and test metadata for embeddings ...")
save_metadata('/tmp/tensorboard/metadata_coarse.tsv', dataset.dataset_coarse_list)
save_metadata('/tmp/tensorboard/metadata_test.tsv', dataset.dataset_test_list)

# create a TensorFlow summary writer
logger.info("Configuring tensorboard embedding ...")
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()

# Coarse Embedding
embedding_conf_coarse = config.embeddings.add()
embedding_conf_coarse.tensor_name = 'embedding_coarse:0'
embedding_conf_coarse.metadata_path = 'metadata_coarse.tsv'

# Test Embedding
embedding_conf_test = config.embeddings.add()
embedding_conf_test.tensor_name = 'embedding_test:0'
embedding_conf_test.metadata_path = 'metadata_test.tsv'

logger.info("Saving embedding ...")
projector.visualize_embeddings(summary_writer, config)

# save the model
logger.info("Saving embedding model ...")
saver = tf.train.Saver()
saver.save(sess, os.path.join('/tmp/tensorboard', "model.ckpt"))