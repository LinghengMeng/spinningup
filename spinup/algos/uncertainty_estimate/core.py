import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, x_dim, y_dim, size):
        self.x_buf = np.zeros([size, x_dim], dtype=np.float32)
        self.y_buf = np.zeros([size, y_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, x, y):
        self.x_buf[self.ptr] = x
        self.y_buf[self.ptr] = y
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        if batch_size > self.size:
            batch_size = self.size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(x=self.x_buf[idxs],
                    y=self.y_buf[idxs])


class BernoulliDropout(tf.keras.layers.Wrapper):
    """
    This wrapper is to implement regularization in BernoulliDropout given by Eq.(4) in https://arxiv.org/abs/1506.02142v6
    # Arguments
        layer: a layer instance
        weight_regularizer:
            a positive number satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset
            Note that kernel_regularizer is not needed, since we will implement it in this wrapper.
        dropout_rate:
            a positive number in (1, 0]
    """
    def __init__(self, layer, weight_regularizer=1e-6, dropout_rate=5e-2, **kwargs):
        assert 'kernel_ragularizer' not in kwargs
        super(BernoulliDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_rate = dropout_rate
        self.retain_rate = 1 - self.dropout_rate
        self.bernoulli_dist = tfp.distributions.Bernoulli(probs=self.retain_rate)

    def build(self, input_shape=None):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise regularizer
        # weight = self.layer.weights
        kernel = self.layer.kernel
        bias = self.layer.bias

        kernel_regularizer = self.weight_regularizer * (1-self.dropout_rate) * tf.reduce_sum(tf.square(kernel))
        bias_regualarizer = self.weight_regularizer * tf.reduce_sum(tf.square(bias))
        regularizer = tf.reduce_sum(kernel_regularizer + bias_regualarizer)
        # Add the regularization loss to collection.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    def conpute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def bernoulli_dropout(self, x):
        """
        Bernoulli dropout - used to mask input
        :param x:
        :return:
        """
        mask = self.bernoulli_dist.sample(sample_shape=tf.shape(x))
        return x*tf.dtypes.cast(mask, tf.float32)

    def call(self, inputs, training=None):
        if training:
            return self.layer.call(self.bernoulli_dropout(inputs))
        else:
            return self.layer.call(inputs)

class ConcreteDropout(tf.keras.layers.Wrapper):
    """This wrapper allows to learn the dropout probability
        for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
             N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eucledian
            loss.

    # Warning
        You must import the actual layer class from tf layers,
         else this will not work.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=(1,),
                                         initializer=tf.keras.initializers.RandomUniform((1,), self.init_min,
                                                                                         self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)

        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.add_to_collection("LAYER_P", self.p)

        # initialise regulariser / prior KL term
        input_dim = int(np.prod(input_shape[-1]))

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - self.p) # TODO: Shouldn't this be time rather than divide??
        dropout_regularizer = self.p * tf.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        # Add the regularisation loss to collection.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = 1e-7
        temp = 0.1

        unif_noise = tf.random_uniform(shape=tf.shape(x))
        drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
        )
        drop_prob = tf.nn.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if training:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            return self.layer.call(inputs)


class MLP(tf.keras.Model):
    """
    Multi-Layer Perceptron Network:
        Model class used to create mlp.
    """
    def __init__(self, layer_sizes=[32],
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None,
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(MLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation=hidden_activation,
                                                            kernel_initializer=kernel_initializer,
                                                            bias_initializer=bias_initializer,
                                                            kernel_regularizer=kernel_regularizer))
        self.out_layer = tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer)

    def call(self, inputs):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        return self.out_layer(x)


class BeroulliDropoutMLP(tf.keras.Model):
    """
    Bernoulli Dropout Multi-Layer Perceptron Network:
        Model class used to create mlp with bernoulli dropout.
    """
    def __init__(self, layer_sizes=[32], weight_regularizer=1e-6, dropout_rate=0.05,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 hidden_activation=tf.keras.activations.relu, output_activation=tf.keras.activations.linear):
        super(BeroulliDropoutMLP, self).__init__()
        self.hidden_layers = []
        for h in layer_sizes[:-1]:
            self.hidden_layers.append(BernoulliDropout(tf.keras.layers.Dense(h,
                                                                             activation=hidden_activation,
                                                                             kernel_initializer=kernel_initializer,
                                                                             bias_initializer=bias_initializer),
                                                       weight_regularizer=weight_regularizer,
                                                       dropout_rate=dropout_rate))
        self.out_layer = BernoulliDropout(tf.keras.layers.Dense(layer_sizes[-1],
                                                                activation=output_activation,
                                                                kernel_initializer=kernel_initializer,
                                                                bias_initializer=bias_initializer),
                                          weight_regularizer=weight_regularizer,
                                          dropout_rate=dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for h_layer in self.hidden_layers:
            x = h_layer(x, training)
        return self.out_layer(x, training)

class BootstrappedEnsemble():
    """
    Bootstrapped Ensemble:
        a set of MLP are trained with their corresponding bootstrapped training set.
    """
    def __init__(self, ensemble_size = 10, x_dim=5, y_dim=2, replay_size=1e6,
                 x_ph=None, y_ph=None, layer_sizes=[], kernel_regularizer=None,
                 hidden_activation=tf.keras.activations.relu,
                 output_activation=tf.keras.activations.sigmoid, learning_rate=1e-3):
        self.ensemble_size = ensemble_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_ph = x_ph
        self.y_ph = y_ph

        self.replay_buffers = []
        self.ensemble = []
        self.ensemble_preds = []
        self.ensemble_losses = []
        self.ensemble_train_ops = []

        for ens_i in range(self.ensemble_size):
            self.replay_buffers.append(ReplayBuffer(x_dim=x_dim, y_dim=y_dim, size=replay_size))
            self.ensemble.append(MLP(layer_sizes, kernel_regularizer=kernel_regularizer,
                                     hidden_activation=hidden_activation,
                                     output_activation=output_activation))
            self.ensemble_preds.append(self.ensemble[ens_i](self.x_ph))
            # mean-square-error
            self.ensemble_losses.append(tf.reduce_mean((self.y_ph - self.ensemble_preds[ens_i]) ** 2))
            self.ensemble_train_ops.append(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.ensemble_losses[ens_i],
                var_list=self.ensemble[ens_i].variables))

    def prediction(self, sess, input):
        # predict
        feed_dict = {self.x_ph: input.reshape(-1, self.x_dim)}
        preds = np.zeros((self.ensemble_size, self.y_dim))
        for i in range(self.ensemble_size):
            preds[i,:] = sess.run(self.ensemble_preds[i], feed_dict=feed_dict)
        return preds

    def add_to_replay_buffer(self, input, label, bootstrapp_p=0.75):
        # add to replay buffer
        for i in range(self.ensemble_size):
            if np.random.uniform(0, 1, 1) < bootstrapp_p:  # with bootstrapp_p probability to add to replay buffer
                self.replay_buffers[i].store(input, label)

    def train(self, sess, raw_batch_size, batch_size, uncertainty_based_minibatch_sample):
        loss = np.zeros((self.ensemble_size,))
        for i in range(self.ensemble_size):
            if uncertainty_based_minibatch_sample:
                # Select top n highest uncertainty samples
                raw_batch = self.replay_buffers[i].sample_batch(raw_batch_size)
                batch = self.resample_based_on_uncertainty(sess, raw_batch, raw_batch_size, batch_size)
            else:
                batch = self.replay_buffers[i].sample_batch(batch_size)
            # Train on the batch
            out = sess.run([self.ensemble_losses[i], self.ensemble_train_ops[i]], feed_dict={self.x_ph: batch['x'], self.y_ph: batch['y']})
            loss[i] = out[0]
        return loss

    def resample_based_on_uncertainty(self, sess, raw_batch, raw_batch_size, batch_size):
        ensemble_postSample = sess.run(self.ensemble_preds, feed_dict={self.x_ph: raw_batch['x'], self.y_ph: raw_batch['y']})
        ensemble_postSample = np.reshape(np.concatenate(ensemble_postSample, axis=1), (-1, self.ensemble_size, self.y_dim))
        uncertainty_rank = np.zeros((raw_batch_size,))
        for i_batch in range(ensemble_postSample.shape[0]):
            uncertainty_rank[i_batch] = np.sum(np.diag(np.cov(ensemble_postSample[i_batch], rowvar=False)))
        # Find top_n highest uncertainty samples
        top_n_highest_uncertainty_indices = np.argsort(-uncertainty_rank)[:batch_size]
        batch = {}
        batch['x'] = raw_batch['x'][top_n_highest_uncertainty_indices]
        batch['y'] = raw_batch['y'][top_n_highest_uncertainty_indices]
        return batch
