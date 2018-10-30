'''
Class representing an EMANN module, a 2-layer fully-connected neural network
with the possibility of adding and pruning units from its hidden layer.

The algorithm used on these modules is an accurate recreation of the original
EMANN algorithm, as described by its creators in the original paper:
SALOME Tristan, BERSINI Hugues,
An Algorithm for Self-Structuring Neural Net Classifiers,
IRIDIA - Universite Libre de Bruxelles, June 28 - July 2 1994.

Modification of Wolfgang Beyer's code from his tutorial:
"Simple Image Classification Models for the CIFAR-10 dataset using TensorFlow".
https://www.wolfib.com/Image-Recognition-Intro-Part-1/
https://www.wolfib.com/Image-Recognition-Intro-Part-2/
'''

# Needed for compatibility between Python 2 and 3.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import relevant modules (numpy, TensorFlow).
import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------
# ---------------------------USEFUL FUNCTIONS---------------------------
# ----------------------------------------------------------------------


def get_activation_funct():
    '''
    Returns the name of the activation function for the hidden layer,
    in this case sigmoid.
    '''
    return "sigmoid"


def connection_strength(weights):
    '''
    The connection strength operation, to be applied on a tensor (vector)
    containing the weights of a unit.

    The CS of a unit is the average of (the absolute values of) its weights.
    '''
    CS = tf.reduce_mean(tf.abs(weights))
    return CS.eval()


def sigmoid_prime(x):
    '''
    The derivative of the sigmoid operation, defined using TensorFlow's
    sigmoid operation.

    sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))
    '''
    return tf.multiply(tf.nn.sigmoid(x),
                       tf.subtract(tf.constant(1.0), tf.nn.sigmoid(x)))

# ----------------------------------------------------------------------
# --------------------------MODULE OBJECT CLASS-------------------------
# ----------------------------------------------------------------------


class Module:
    '''Defines a module, a 2-layer fully-connected neural network'''

    def __init__(self, module_ID, input_channel, input_units, hidden_units,
                 classes):
        '''
        Initialization method.
        Sets the initial weights and biases for each layer in the network.
        Defines the forward pass through the module.

        Args:
            module_ID: Identifier for the module.
            input_channel: Input data placeholder.
            input_units: Number of input entry units (i.e. color channels).
            hidden_units: Number of initial hidden units.
            classes: Number of image classes (number of possible labels).

        Returns:
            None.
        '''
        # The module's ID is used when defining TensorFlow variables.
        self.module_ID = module_ID

        # Define the initial number of units for each layer.
        self.a_0_units = input_units
        self.a_1_units = hidden_units
        self.a_2_units = classes

        # Variables for the number of settled units, and the min and max CS.
        self.settled = 0
        self.minCS = 0
        self.maxCS = 0

        # ------------------LAYER #1------------------
        # Define the layer's weights as a list of vectors.
        # N.B.: The matrix used for backpropagation is found by stacking
        #       these vectors along axis=1 (transpose of just staking them).
        self.w_1 = []
        for unit in range(self.a_1_units):
            self.w_1.append(tf.get_variable(
                name='m_'+str(self.module_ID)+'__w_1_'+str(unit+1),
                shape=[self.a_0_units],
                # Weights are initialized to normally distributed variables,
                # the standard deviation being 1/sqrt(a_0_units).
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(float(self.a_0_units)))
            ))

        # ------------------LAYER #2------------------
        # Define the layer's weights as a list of vectors.
        # N.B.: The matrix used for backpropagation is found by stacking
        #       these vectors along axis=0 (just staking them).
        self.w_2 = []
        for unit in range(self.a_1_units):
            self.w_2.append(tf.get_variable(
                name='m_'+str(self.module_ID)+'__w_2_'+str(unit+1),
                shape=[self.a_2_units],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(float(self.a_1_units)))
            ))

        # -----------DEFINE THE FORWARD PASS----------
        # Keep a reference to the input channels.
        self.a_0 = input_channel
        # Define the hidden units' output (w.r.t. input units and weights).
        # N.B.: The activation function is sigmoid (value between 0 and 1).
        self.z_1 = tf.matmul(self.a_0, tf.stack(self.w_1, axis=1))
        self.a_1 = tf.nn.sigmoid(self.z_1)
        # Define the output units' output (w.r.t. hidden units and weights).
        # N.B.: The activation function is also the sigmoid.
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.sigmoid(self.z_2)

    def add_new_unit(self, sess, unit_count):
        '''
        Adds new units to the module's hidden layer (new weight vectors).
        Then redefines the forward pass to take the new unit into account.

        Args:
            sess: Ongoing TensorFlow session.
            unit_count: Number of new units to be added.

        Returns:
            None.
        '''
        # For each unit to be added.
        for unit in range(unit_count):
            # Increase the number of units in the hidden layer.
            self.w_1.append(tf.get_variable(
                name='m_'+str(self.module_ID)+'__w_1_'+str(self.a_1_units+1),
                shape=[self.a_0_units],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(float(self.a_0_units)))
            ))
            self.w_2.append(tf.get_variable(
                name='m_'+str(self.module_ID)+'__w_2_'+str(self.a_1_units+1),
                shape=[self.a_2_units],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(float(self.a_1_units)))
            ))
            # Initializes these new weight vectors.
            sess.run(self.w_1[-1].initializer)
            sess.run(self.w_2[-1].initializer)
            # Update the unit count.
            self.a_1_units += 1

        # Redefine the forward pass accordingly.
        self.z_1 = tf.matmul(self.a_0, tf.stack(self.w_1, axis=1))
        self.a_1 = tf.nn.sigmoid(self.z_1)
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.sigmoid(self.z_2)

        print("Added an unit! Number of units:", self.a_1_units)

    def units_settled(self, settling_tresh, extra_cond=False):
        '''
        Checks how many units in the module's hidden layer have settled
        (their connection strength is above a settling treshold).

        An additional condition for settling can be added, where the unit
        must have at least one of its weights above the settling treshold.
        This condition is used to avoid "copy" problems in succesive modules.

        Args:
            settling_tresh: The settling treshold for CS or weights.
            extra_cond: Adds an extra condition for weights (default False).

        Returns:
            settled: The number of units that have settled.
        '''
        self.maxCS = connection_strength(self.w_1[0])
        self.minCS = connection_strength(self.w_1[0])
        self.settled = 0

        # Check for every unit if the CS is above the threshold.
        for i in range(self.a_1_units):
            CS = connection_strength(self.w_1[i])
            self.maxCS = max(CS, self.maxCS)
            self.minCS = min(CS, self.minCS)
            if(CS > settling_tresh):
                if(extra_cond):
                    # Check for every weight if it is above the threshold.
                    current_weights = self.w_1[i].eval()
                    for j in range(len(current_weights)):
                        if(current_weights[j] > settling_tresh):
                            self.settled += 1
                else:
                    self.settled += 1

        print("CS from", self.minCS, "to", self.maxCS,
              "Settling tresh:", settling_tresh)
        print("Settled units:", self.settled, "\n")
        return self.settled

    def units_prune(self, uselessness_tresh):
        '''
        Prunes all the useless units in the hidden layer (their connection
        strength is below an uselessness treshold).

        Args:
            uselessness_tresh: The uselessness treshold for CS.

        Returns:
            None.
        '''
        toKeep = []
        # Check for every unit if the CS is above the threshold.
        for i in range(self.a_1_units):
            CS = connection_strength(self.w_1[i])
            if(CS > uselessness_tresh):
                toKeep.append(i)  # If so, keep the unit.

        print("Number of units before pruning:", self.a_1_units)
        print("Units to prune:", self.a_1_units - len(toKeep))

        # Update the number of units.
        self.a_1_units = len(toKeep)
        # Recreate the weight vector lists without the pruned units.
        self.w_1 = [self.w_1[i] for i in toKeep]
        self.w_2 = [self.w_2[i] for i in toKeep]

        # Redefine the forward pass after pruning.
        self.z_1 = tf.matmul(self.a_0, tf.stack(self.w_1, axis=1))
        self.a_1 = tf.nn.sigmoid(self.z_1)
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.sigmoid(self.z_2)

        print("Pruned units! Number of units:", self.a_1_units)

    def training(self, y, eta):
        '''
        Define what to do in each training epoch (backward propagation).

        Args:
            y: Labels tensor, of type int64 - [batch size].
            eta: Learning rate to use for backward propagation.

        Returns:
            train_epoch: Operation for a training epoch.
        '''
        # Difference between probabilities and actual labels.
        # N.B.: One-hot encoding converts the labels into probabilities.
        diff = tf.subtract(self.a_2, tf.one_hot(y, self.a_2_units))
        # Difference between old and new values for weights.
        # Cost function is the square difference.
        cost = tf.multiply(diff, diff)

        # Use the cost to update the weights, modulated by the learning rate.
        train_epoch = tf.train.GradientDescentOptimizer(eta).minimize(cost)

        return train_epoch

    def evaluation(self, y):
        '''
        Define an operation for calculating the accuracy of predictions,
        how well the probabilities can predict the image's actual label.

        This operation is the proportion of correct predictions, the mean
        of the results of the binary test "does the component of a_2 with
        the greatest value correspond to the expected output y?".

        Args:
            y: Labels tensor, of type int64 - [batch size].

        Returns:
            accuracy: Percentage of images whose class was correctly predicted.
        '''
        # Define an operation for comparing the prediction with the true label.
        correct_prediction = tf.equal(tf.argmax(self.a_2, 1), y)

        # Define an operation for calculating the accuracy of predictions.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def quadr_error(self, y):
        '''
        Define an operation for calculating the mean square error (MSE)
        of the predicted labels.

        Args:
            y: Labels tensor, of type int64 - [batch size].

        Returns:
            quadr_error: Mean square error of the predictions.
        '''
        quadr_error = tf.losses.mean_squared_error(
            labels=tf.one_hot(y, self.a_2_units), predictions=self.a_2)

        return quadr_error
