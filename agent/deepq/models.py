import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)

from common.tf_util import ortho_init
def _lstm_to_mlp(cell,max_length,aktiv,hiddens,dueling, inpt,seq, num_actions, scope, reuse=False, layer_norm=False,init_scale=1.0):
    nh,nin,nfeatures = cell
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("lstmnet"):
            wx  = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
            wh  = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
            wh0  = tf.get_variable("wh0", [nfeatures, nh], initializer=ortho_init(init_scale))
            b   = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))
            c   = tf.zeros(shape=(tf.shape(inpt)[0],nh),dtype=tf.float32)
            h = tf.matmul(inpt[:,0,-nfeatures:],wh0)
            out = []
            for idx in range(max_length):
                x = inpt[:,idx,:nin];
                z = tf.matmul( tf.zeros(shape=tf.shape(x)) if idx==0 else x, wx) + tf.matmul(h, wh) +  b
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i)
                f = tf.nn.sigmoid(f)
                o = tf.nn.sigmoid(o)
                u = tf.tanh(u)
                c = f*c + i*u
                h = o*tf.tanh(c)
                out.append(h)
            out     = tf.transpose(tf.stack(out),perm=[1,0,2])
#            return xs, s
#            out              = tf.unstack(inpt, max_length, 1)
#            out, states     = tf.nn.static_rnn(tf.nn.rnn_cell.LSTMCell(lstm_cells), out, dtype=tf.float32,sequence_length=seq)
#            out             = tf.stack(out)
            batch_size      = tf.shape(out)[0]
            index           = tf.range(0, batch_size) * max_length + (seq - 1)
            out             = tf.gather(tf.reshape(out, [-1, nh]), index)
            lstm_out        = layers.fully_connected(out, num_outputs=nh, activation_fn=aktiv)
        with tf.variable_scope("action_value"):
            action_out = lstm_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = aktiv(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = lstm_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out
    
def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def lstm_to_mlp(cell,max_length,aktiv,hiddens, dueling=False, layer_norm=False):
    return lambda *args, **kwargs: _lstm_to_mlp(cell,max_length,aktiv,hiddens,dueling, layer_norm=layer_norm, *args, **kwargs)

def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

