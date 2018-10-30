'''
Trains and evaluates an EMANN classifier for CIFAR-10.

Modification of Wolfgang Beyer's code form his tutorial:
"Simple Image Classification Models for the CIFAR-10 dataset using TensorFlow".
'''

# Needed for compatibility between Python 2 and 3.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import relevant modules (numpy, TensorFlow, time modules, OS file paths).
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import sys
# data_helpers.py contains functions for loading and preparing the dataset.
import data_helpers
# two_layer_fc.py contains the Module class, which defines a 2-layer NN.
from EMANN_softmax import Module, connection_strength, get_activation_funct

# Silence TensorFlow warning logs on build.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('--------------------------------------------------------------')
print('----- CIFAR-10 TensorFlow EMANN classifier training demo -----')
print('--------------------------------------------------------------\n')


# -----------------------------------------------------------------------------
# ----------------------------CONSTANT DEFINITIONS-----------------------------
# -----------------------------------------------------------------------------


# Define as constants: number of color channels per image, number of classes.
IMAGE_PIXELS = 3072
CLASSES = 10

# Define some parameters for the model, as external TensorFlow flags.
flags = tf.flags
FLAGS = flags.FLAGS
# Parameters related to training epochs.
flags.DEFINE_integer('batch_size', 400, 'Batch size, divisor of dataset size.')
flags.DEFINE_float('learning_rate', 1e-4,
                   'Learning rate for the training epoch (backpropagation).')
# Parameters used in EMANN algorithms (with alt values).
flags.DEFINE_float('uselessness_tresh', 0.15,
                   'Minimum connection strength for a unit to be useless.')
flags.DEFINE_float('settling_tresh', 0.5,
                   'Maximum connection strength for a unit to be settled.')
flags.DEFINE_integer('ascension_tresh', 200,
                     'During ascension, number of epochs for adding a unit.')
flags.DEFINE_float('uselessness_tresh_alt', 0.013,
                   'Different uselessness treshold for the first module.')
flags.DEFINE_float('settling_tresh_alt', 0.016,
                   'Different settling treshold for the first module.')
flags.DEFINE_integer('ascension_tresh_alt', 200,
                     'Different ascension treshold for the first module.')
# Parameters used in EMANN algorithms (without alt values).
flags.DEFINE_integer('initial_unit_num', 1,
                     'Initial number of units in the first module.')
flags.DEFINE_integer('unit_increase', 1,
                     'Number of units added each ascension_thresh.')
flags.DEFINE_integer('ascension_limit', 120,
                     'Maximum number of neurons before stopping ascension.')
flags.DEFINE_integer('patience_param', 2000,
                     'During improvement, number of epochs before stopping.')
flags.DEFINE_float('improvement_tresh', 5e-3,
                   'Minimum MSE difference for adding a new module.')
# Parameters related to the algorithm's implementation.
flags.DEFINE_integer('check_tresh', 100,
                     'Number of epochs to skip before the network is checked.')
flags.DEFINE_string('train_dir', 'tf_logs',
                    'Directory for the program to place the training logs.')

FLAGS(sys.argv)
print('\n-------SELECTED VALUES FOR NN PARAMETERS-------')
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print('{} = {}'.format(attr, value))
print()


# -----------------------------------------------------------------------------
# ----------------------------FUNCTION DEFINITIONS-----------------------------
# -----------------------------------------------------------------------------


def get_next_feed_dict(batches):
    '''
    Get a random input data batch for the next training epoch.

    Args:
        batches: A generator that yields randomly generated batches.

    Returns:
        feed_dict: A dictionary that defines the next data batch for training.
    '''
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
        images_placeholder: images_batch,
        labels_placeholder: labels_batch
    }
    return feed_dict


def standard_module_check(sess, new_module, accuracy, feed_dict,
                          test_feed_dict, num_epochs, logs):
    '''
    Standard check operation performed on the module every check_tresh epochs.
    Calculates the accuracy, the maximal and minimal CS, and adds the epoch
    data to the custom training log file.

    Args:
        sess: Ongoing TensorFlow session.
        accuracy: Operation for the module's accuracy calculation.
        feed_dict: Random input data batch for the current epoch.
        test_feed_dict: Input data batch from the testing set.
        num_epochs: Number of training epochs already elapsed.
        logs: A writer for custom training log files.

    Returns:
        train_accuracy: The current accuracy of the module on the training set.
    '''

    # Calculate the accuracy on the training set.
    train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
    # Print the calculated training set accuracy.
    print('Epoch {:d}: training set accuracy {:g}'.format(
            num_epochs, train_accuracy))
    logs.write('{:d}\t{:d}\t{:d}\t{:g}\t{:g}\t{:g}\t{:g}\n'.format(
        num_epochs, new_module.a_1_units, new_module.settled,
        train_accuracy, test_accuracy, new_module.minCS, new_module.maxCS))

    return train_accuracy


def EMANN_ascension(sess, new_module, labels_placeholder, batches,
                    test_feed_dict, units_settled, num_epochs, logs):
    '''
    Run the ascension stage of EMANN on a module.

    Returns:
        None.
    '''
    # From the module, get definitions for the training epoch
    # and the accuracy calculation.
    train_epoch = new_module.training(labels_placeholder, FLAGS.learning_rate)
    accuracy = new_module.evaluation(labels_placeholder)

    # Get the proper tresholds for the module.
    settling_tresh = FLAGS.settling_tresh
    ascension_tresh = FLAGS.ascension_tresh
    if new_module.module_ID == 0:
        settling_tresh = FLAGS.settling_tresh_alt
        ascension_tresh = FLAGS.ascension_tresh_alt

    # Perform training epochs until one unit settles
    # or the hidden layer has reached ascension_limit units.
    # Add a new unit every ascension_tresh epochs.
    while units_settled == 0:  # and new_module.a_1_units < FLAGS.ascension_limit:
        # Get the (random) input data batch corresponding to epoch i.
        feed_dict = get_next_feed_dict(batches)
        # Perform a new training epoch.
        sess.run(train_epoch, feed_dict=feed_dict)

        # Every ascension_tresh epochs, add unit_increase units to the hidden layer.
        if num_epochs % ascension_tresh == 0:
            # Add the new unit and update the graph.
            new_module.add_new_unit(sess, FLAGS.unit_increase)
            # Update the definitions of training epoch and accuracy.
            train_epoch = new_module.training(labels_placeholder,
                                              FLAGS.learning_rate)
            accuracy = new_module.evaluation(labels_placeholder)

        # Every check_tresh epochs, check the state of the module.
        if num_epochs % FLAGS.check_tresh == 0:
            # Check if a unit has settled.
            units_settled = new_module.units_settled(settling_tresh)

            # Perform the standard module check.
            train_accuracy = standard_module_check(sess, new_module,
                                                   accuracy, feed_dict,
                                                   test_feed_dict,
                                                   num_epochs, logs)

        # Increase the epoch counter.
        num_epochs += 1

    return units_settled, num_epochs


def EMANN_improvement(sess, new_module, labels_placeholder, batches,
                      test_feed_dict, units_settled, num_epochs, logs):
    '''
    Run the improvement stage of EMANN on a module.

    Returns:
        None.
    '''
    # From the module, get definitions for the training epoch
    # and the accuracy calculation.
    train_epoch = new_module.training(labels_placeholder, FLAGS.learning_rate)
    accuracy = new_module.evaluation(labels_placeholder)

    # Get the proper settling treshold for the module.
    settling_tresh = FLAGS.settling_tresh
    if new_module.module_ID == 0:
        settling_tresh = FLAGS.settling_tresh_alt

    # Perform training epochs until no units have settled in awhile.
    # Add a new unit every time a unit settles.
    patience_countdown = FLAGS.patience_param
    while patience_countdown >= 0:
        # Get the (random) input data batch corresponding to epoch i.
        feed_dict = get_next_feed_dict(batches)
        # Perform a new training epoch.
        sess.run(train_epoch, feed_dict=feed_dict)

        # Every check_tresh epochs, check the state of the module.
        if num_epochs % FLAGS.check_tresh == 0:
            # Check if new units have settled, if so add a new unit.
            new_units_settled = new_module.units_settled(settling_tresh)
            if new_units_settled > units_settled:
                # Update the number of settled units.
                units_settled = new_units_settled
                # Reset the patience countdown.
                patience_countdown = FLAGS.patience_param + 1

                # Add the new unit and update the graph.
                new_module.add_new_unit(sess, 1)
                # Update the definitions of training epoch and accuracy.
                train_epoch = new_module.training(labels_placeholder,
                                                  FLAGS.learning_rate)
                accuracy = new_module.evaluation(labels_placeholder)

            # Perform the standard module check.
            train_accuracy = standard_module_check(sess, new_module,
                                                   accuracy, feed_dict,
                                                   test_feed_dict,
                                                   num_epochs, logs)

        # Increase the epoch counter and decrease the patience countdown.
        num_epochs += 1
        patience_countdown -= 1

    return units_settled, num_epochs


def EMANN_module_training(sess, module_ID, inputs_placeholder, input_num,
                          initial_unit_num, labels_placeholder, batches,
                          test_feed_dict, logs):
    '''
    Run the EMANN algorithm to train a module (network with one hidden layer).

    Args:
        sess: Ongoing TensorFlow session.
        module_ID: Identifier for the module to be built.
        inputs_placeholder: Represents the inputs for this module.
        input_num: Number of input channels for this module.
        initial_unit_num: Initial number of units in the hidden layer
        labels_placeholder: Represents the expected outputs for this module.
        batches: A generator of random data batches used for training.
        test_feed_dict: A spare data batch used for testing.
        logs: A writer for custom training log files.

    Returns:
        new_module: A new module to add to the EMANN classifier.
    '''
    print('Setting up the new module...', end='')
    # Create the module, with initial_unit_num units in its hidden layer.
    new_module = Module(module_ID, inputs_placeholder, input_num,
                        initial_unit_num, CLASSES)
    # From the module, get definitions for the training epoch
    # and the accuracy calculation.
    train_epoch = new_module.training(labels_placeholder, FLAGS.learning_rate)
    accuracy = new_module.evaluation(labels_placeholder)
    print(' DONE!')

    # Get the proper settling threshold (useful for the standard check).
    settling_tresh = FLAGS.settling_tresh
    if new_module.module_ID == 0:
        settling_tresh = FLAGS.settling_tresh_alt

    # Initialize global variables.
    sess.run(tf.global_variables_initializer())
    # Current number of settled units is 0.
    settled_units = 0
    # Current number of performed epochs is 0.
    num_epochs = 0

    # Get the (random) input data batch corresponding to epoch i.
    feed_dict = get_next_feed_dict(batches)
    # Perform the first training epoch and increase the epoch counter.
    sess.run(train_epoch, feed_dict=feed_dict)
    units_settled = new_module.units_settled(settling_tresh)
    num_epochs += 1

    # ------------------------ ASCENSION STAGE ------------------------
    print('\n--------------1. ASCENSION STAGE---------------\n')
    logs.write('--------------1. ASCENSION STAGE---------------\n')
    logs.write('Epoch\tUnits\tSettl\tAccTr\tAccTes\tCS Min\tCS Max\n')
    units_settled, num_epochs = EMANN_ascension(sess, new_module,
                                                labels_placeholder, batches,
                                                test_feed_dict,
                                                units_settled, num_epochs,
                                                logs)

    # ----------------------- IMPROVEMENT STAGE -----------------------
    print('\n-------------2. IMPROVEMENT STAGE--------------\n')
    logs.write('-------------2. IMPROVEMENT STAGE--------------\n')
    logs.write('Epoch\tUnits\tSettl\tAccTr\tAccTes\tCS Min\tCS Max\n')
    units_settled, num_epochs = EMANN_improvement(sess, new_module,
                                                  labels_placeholder, batches,
                                                  test_feed_dict,
                                                  units_settled, num_epochs,
                                                  logs)

    # ------------------------- PRUNING STAGE -------------------------
    print('\n---------------3. PRUNING STAGE----------------\n')
    logs.write('---------------3. PRUNING STAGE----------------\n')
    logs.write('Epoch\tUnits\tSettl\tAccTr\tAccTes\tCS Min\tCS Max\n')
    # Save the current accuracy.
    pre_pruning_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
    units_settled = new_module.units_settled(settling_tresh)
    print('Epoch {:d}: training set accuracy BEFORE PRUNING {:g}'.format(
            num_epochs, pre_pruning_accuracy))
    logs.write('{:d}\t{:d}\t{:d}\t{:g}\t{:g}\t{:g}\t{:g}\n'.format(
        num_epochs, new_module.a_1_units, new_module.settled,
        pre_pruning_accuracy, test_accuracy, new_module.minCS, new_module.maxCS))

    # Prune all useless units.
    if new_module.module_ID == 0:
        new_module.units_prune(FLAGS.uselessness_tresh_alt)
    else:
        new_module.units_prune(FLAGS.uselessness_tresh)
    # Update the definitions of training epoch and accuracy.
    train_epoch = new_module.training(labels_placeholder, FLAGS.learning_rate)
    accuracy = new_module.evaluation(labels_placeholder)

    # Save the accuracy after pruning.
    train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
    units_settled = new_module.units_settled(settling_tresh)
    print('Epoch {:d}: training set accuracy AFTER PRUNING {:g}'.format(
            num_epochs, train_accuracy))
    logs.write('{:d}\t{:d}\t{:d}\t{:g}\t{:g}\t{:g}\t{:g}\n'.format(
        num_epochs, new_module.a_1_units, new_module.settled,
        train_accuracy, test_accuracy, new_module.minCS, new_module.maxCS))

    # ------------------------ RECOVERY STAGE -------------------------
    print('\n--------------4. RECOVERY STAGE----------------\n')
    logs.write('--------------4. RECOVERY STAGE----------------\n')
    logs.write('Epoch\tUnits\tSettl\tAccTr\tAccTes\tCS Min\tCS Max\n')
    # Perform training epochs until the accuracy is the same as before pruning.
    while train_accuracy < pre_pruning_accuracy:
        # Get the (random) input data batch corresponding to epoch i.
        batch = next(batches)
        images_batch, labels_batch = zip(*batch)
        feed_dict = get_next_feed_dict(batches)
        # Perform a new training epoch.
        sess.run(train_epoch, feed_dict=feed_dict)

        # Every check_tresh epochs, check the state of the module.
        if num_epochs % FLAGS.check_tresh == 0:
            # Perform the standard module check.
            units_settled = new_module.units_settled(settling_tresh)
            train_accuracy = standard_module_check(sess, new_module,
                                                   accuracy, feed_dict,
                                                   test_feed_dict,
                                                   num_epochs, logs)

        # Increase the epoch counter.
        num_epochs += 1

    # Perform one last standard module check.
    units_settled = new_module.units_settled(settling_tresh)
    train_accuracy = standard_module_check(sess, new_module, accuracy,
                                           feed_dict, test_feed_dict,
                                           num_epochs, logs)

    test_accuracy = sess.run(accuracy, feed_dict=test_feed_dict)
    print('--------------------------------------------------------------')
    print('\aFINAL TESTING SET ACCURACY {:g}'.format(test_accuracy))
    print('--------------------------------------------------------------')

    return new_module


# -----------------------------------------------------------------------------
# ------------------------------MAIN PROGRAM CODE------------------------------
# -----------------------------------------------------------------------------


# Start measuring the runtime.
beginTime = time.time()

print('Loading data sets...', end='')

# Load the CIFAR-10 data (see data_helpers.py).
# 50000 training set images, 10000 testing set images.
# Data sets are: images_train, labels_train, images_test, labels_test, classes.
data_sets = data_helpers.load_data()

print(' DONE!')

print('Creating placeholders for datasets and modules...', end='')

# Define input placeholders. An input contains:
# 1) a batch of "N" images, where each is a batch of 3072 floats (1024 pixels),
# 2) a batch of "N" labels, where each is an int (from 0 to 9).
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
                                    name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None],
                                    name='image-labels')
# Initialize the list of modules.
modules_list = []
# Initialize the list of inputs for modules.
input_list = [images_placeholder]
input_num = IMAGE_PIXELS

print(' DONE!')

print('\n---------EMANN TRAINING SESSION BEGINS---------\n')

with tf.Session() as sess:
    print('Generating {:d} random batches...'.format(FLAGS.batch_size), end='')
    # Build a generator of random input data batches, of size batch_size.
    zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
    batches = data_helpers.generate_random_batches(
        list(zipped_data), FLAGS.batch_size)
    print(' DONE!')
    # Retrieve the test data batch to be used after training each module.
    test_feed_dict = {
            images_placeholder: data_sets['images_test'],
            labels_placeholder: data_sets['labels_test']
    }

    # Put custom training logs in a .txt file.
    date_and_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    logfile = FLAGS.train_dir + '/' + get_activation_funct()
    logfile = logfile + '__' + date_and_time + '.txt'
    # Create a file writer for custom log files.
    log_writer = open(logfile, 'w')
    log_writer.write('Parameters (1st layer):\n')
    log_writer.write('ascension_tresh = {}\n'.format(
        FLAGS.ascension_tresh_alt))
    log_writer.write('settling_tresh = {}\n'.format(FLAGS.settling_tresh_alt))
    log_writer.write('uselessness_tresh = {}\n'.format(
        FLAGS.uselessness_tresh_alt))
    log_writer.write('\nParameters (other layers):\n')
    log_writer.write('ascension_tresh = {}\n'.format(FLAGS.ascension_tresh))
    log_writer.write('settling_tresh = {}\n'.format(FLAGS.settling_tresh))
    log_writer.write('uselessness_tresh = {}\n'.format(
        FLAGS.uselessness_tresh))
    log_writer.write('\nParameters (all layers):\n')
    log_writer.write('patience_param = {}\n'.format(
        FLAGS.patience_param))

    # Itteratively build a certain number of modules (layers).
    initial_unit_num = FLAGS.initial_unit_num
    for i in range(3):
        # Build a new module and add it to the modules list.
        print('Building module number: ', len(modules_list))
        log_writer.write('\nMODULE NUMBER {}\n'.format(len(modules_list)))
        new_module = EMANN_module_training(sess, len(modules_list),
                                           input_list[-1], input_num,
                                           initial_unit_num,
                                           labels_placeholder,
                                           batches, test_feed_dict,
                                           log_writer)
        modules_list.append(new_module)
        # New modules connect to the network through the previous hidden layer.
        input_list.append(new_module.a_1)
        input_num = new_module.a_1_units
        # New modules start with as many hidden units as the previous module.
        initial_unit_num = new_module.a_1_units

    # After all EMANN modules has finished training, the training stops.
    print('\n---------END OF EMANN TRAINING SESSION---------\n')
    # Close the custom training log file.
    log_writer.close()

endTime = time.time()
print('Total time elapsed: {:5.2f}s'.format(endTime - beginTime))
