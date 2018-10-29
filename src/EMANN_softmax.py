'''
Class representing an EMANN module, a 2-layer fully-connected neural network
with the possibility of adding and pruning units from its hidden layer.

The algorithm used on these modules is mostly a recreation of the original
EMANN algorithm, except it uses the softmax function as an
activation function (instead of sigmoid) for all layers.

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
    in this case softmax.
    '''
    return "softmax"


def connection_strength(weights):
    '''
    The connection strength operation, to be applied on a tensor (vector)
    containing the weights of a unit.

    The CS of a unit is the average of (the absolute values of) its weights.
    '''
    CS = tf.reduce_mean(tf.abs(weights))
    return CS.eval()


def softmax_prime(x):
    '''
    The derivative of the softmax operation, defined using TensorFlow's
    softmax operation. It is calculated similarly to the sigmoid operation's
    derivative, because of the quotient rule.

    softmax_prime(x) = softmax(x) * (1 - softmax(x))
    '''
    return tf.multiply(tf.nn.softmax(x),
                       tf.subtract(tf.constant(1.0), tf.nn.softmax(x)))


# ----------------------------------------------------------------------
# --------------------------MODULE OBJECT CLASS-------------------------
# ----------------------------------------------------------------------


class Module:
    '''Defines a module, a 2-layer fully-connected neural network'''

    def __init__(self, module_ID, input_channel, input_units, hidden_units,
                 classes, reg_constant=0):
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
            reg_constant: Regularization constant (default 0).

        Returns:
            None.
        '''
        # The module's ID is used when defining TensorFlow variables.
        self.module_ID = module_ID

        # Define the initial number of units for each layer.
        self.a_0_units = input_units
        self.a_1_units = hidden_units
        self.a_2_units = classes

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
                    stddev=1.0 / np.sqrt(float(self.a_0_units))),
                # L2-regularization adds the sum of the squares of all the
                # weights in the network to the loss function. The importance
                # of this effect is controlled by reg_constant.
                regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
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
                    stddev=1.0 / np.sqrt(float(self.a_1_units))),
                regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
            ))

        # -----------DEFINE THE FORWARD PASS----------
        # Keep a reference to the input channels.
        self.a_0 = input_channel
        # Define the hidden units' output (w.r.t. input units and weights).
        # N.B.: The activation function is softmax (value between 0 and 1).
        self.z_1 = tf.matmul(self.a_0, tf.stack(self.w_1, axis=1))
        self.a_1 = tf.nn.softmax(self.z_1)
        # Define the output units' output (w.r.t. hidden units and weights).
        # N.B.: The activation function is also the softmax.
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.softmax(self.z_2)

    def add_new_unit(self, sess, unit_count, reg_constant=0):
        '''
        Adds new units to the module's hidden layer (new weight vectors).
        Then redefines the forward pass to take the new unit into account.

        Args:
            sess: Ongoing TensorFlow session.
            unit_count: Number of new units to be added.
            reg_constant: Regularization constant (default 0).

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
                    stddev=1.0 / np.sqrt(float(self.a_0_units))),
                regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
            ))
            self.w_2.append(tf.get_variable(
                name='m_'+str(self.module_ID)+'__w_2_'+str(self.a_1_units+1),
                shape=[self.a_2_units],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / np.sqrt(float(self.a_1_units))),
                regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
            ))
            # Initializes these new weight vectors.
            sess.run(self.w_1[-1].initializer)
            sess.run(self.w_2[-1].initializer)
            # Update the unit count.
            self.a_1_units += 1

        # Redefine the forward pass accordingly.
        self.z_1 = tf.matmul(self.a_0, tf.stack(self.w_1, axis=1))
        self.a_1 = tf.nn.softmax(self.z_1)
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.softmax(self.z_2)

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
        self.a_1 = tf.nn.softmax(self.z_1)
        self.z_2 = tf.matmul(self.a_1, tf.stack(self.w_2, axis=0))
        self.a_2 = tf.nn.softmax(self.z_2)

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
        # We use the sigmoid prime function to backpropagate the difference
        # between a_2 and y, and deduce how to alter all the variables.

        # ------------------LAYER #2------------------
        # Difference between probabilities and actual labels.
        # N.B.: One-hot encoding converts the labels into probabilities.
        diff = tf.subtract(self.a_2, tf.one_hot(y, self.a_2_units))
        # Difference between old and new values for weights.
        d_z_2 = tf.multiply(diff, softmax_prime(self.z_2))
        d_w_2 = tf.unstack(tf.matmul(tf.transpose(self.a_1), d_z_2), axis=0)

        # ------------------LAYER #1------------------
        # Difference between hidden layer values and its expected values.
        d_a_1 = tf.matmul(d_z_2, tf.transpose(tf.stack(self.w_2, axis=0)))
        # Difference between old and new values for weights.
        d_z_1 = tf.multiply(d_a_1, softmax_prime(self.z_1))
        d_w_1 = tf.unstack(tf.matmul(tf.transpose(self.a_0), d_z_1), axis=1)

        # Use the deduced differences to update the weights.
        # N.B.: The differences are in fact modulated by the learning rate.
        train_epoch = []
        for unit in range(self.a_1_units):
            train_epoch.append(tf.assign(self.w_1[unit], tf.subtract(
                self.w_1[unit], tf.multiply(eta, d_w_1[unit]))))
            train_epoch.append(tf.assign(self.w_2[unit], tf.subtract(
                self.w_2[unit], tf.multiply(eta, d_w_2[unit]))))

        return train_epoch

    def loss(self, y):
        '''
        Define how the loss value is calculated from the logits (z_2)
        and the actual labels (y).
        N.B: Cross entropy compares the "score ranges" probability
             distribution with the actual distribution
             (correct class = 1, others = 0).

        Args:
            y: Labels tensor, of type int64 - [batch size].

        Returns:
            loss: Loss tensor, of type float32.
        '''
        with tf.name_scope('Loss'):
            # Cross entropy operation between logits and labels.
            # N.B.: Use z_2 because a_2 already is a softmax (of z_2).
            # N.B.2: One-hot encoding converts the labels into probabilities.
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.z_2, labels=tf.one_hot(y, self.a_2_units),
                    name='cross_entropy'))

            # Operation for the final loss function.
            # N.B.: We add all the L2-regularization terms
            #       (squares of weights, multiplied by the reg_constant).
            loss = cross_entropy + tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))

        return loss

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
