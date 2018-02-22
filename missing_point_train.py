import random
import numpy as np
import tensorflow as tf
from ase.db import connect
import tensorfieldnetworks.layers as layers
import tensorfieldnetworks.utils as utils
from tensorfieldnetworks.utils import EPSILON, FLOAT_TYPE


def atom_type_to_one_hot(atom_numbers, atom_order):
    one_hot_dict = {atom_type: [1 if i == j else 0 for i in range(len(atom_order))]
                    for j, atom_type in enumerate(atom_order)}
    return list(map(lambda x: one_hot_dict[x], atom_numbers))


training_set_size = 1000

with connect('qm9.db') as conn:
    qm9_coords = []
    qm9_atoms = []
    qm9_test_coords = []
    qm9_test_atoms = []
    for atoms in conn.select('4<natoms<=18', limit=training_set_size):
        qm9_coords.append(atoms.positions)
        qm9_atoms.append(atoms.numbers)
    for atoms in conn.select('natoms=19', limit=training_set_size):
        qm9_test_coords.append(atoms.positions)
        qm9_test_atoms.append(atoms.numbers)

atom_order = list(set(np.concatenate(qm9_atoms)))
num_atom_types = len(atom_order)

qm9_one_hot = list(map(lambda x: atom_type_to_one_hot(x, atom_order), qm9_atoms))
qm9_test_one_hot = list(map(lambda x: atom_type_to_one_hot(x, atom_order), qm9_test_atoms))

# BUILD NETWORK
print("Building graph.")

# radial basis functions
rbf_low = 0.
rbf_high = 2.5
rbf_count = 4
rbf_spacing = (rbf_high - rbf_low) / rbf_count
centers = tf.cast(tf.lin_space(rbf_low, rbf_high, rbf_count), FLOAT_TYPE)

# r : [N, 3]
r = tf.placeholder(FLOAT_TYPE, shape=(None, 3))

# one_hot : [N, num_atom_types]
one_hot = tf.placeholder(FLOAT_TYPE, shape=(None, num_atom_types))

# [N, N, 3]
rij = utils.difference_matrix(r)

# [N, N, 3]
unit_vectors = rij / tf.expand_dims(tf.norm(rij, axis=-1) + EPSILON, axis=-1)

dij = utils.distance_matrix(r)

# rbf : [N, N, rbf_count]
gamma = 1. / rbf_spacing
rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - centers))

layer_dims = [15, 15, 15, 1]

# EMBEDDING
# [N, layer1_dim, 1]
with tf.variable_scope(None, 'embed', values=[one_hot]):
    embed = layers.self_interaction_layer_with_biases(tf.reshape(one_hot, [-1, num_atom_types, 1]), layer_dims[0])
    input_tensor_list = {0: [embed]}

# LAYERS 1-3
num_layers = len(layer_dims) - 1
for layer in range(num_layers):
    layer_dim = layer_dims[layer + 1]
    with tf.variable_scope(None, 'layer' + str(layer), values=[input_tensor_list]):
        input_tensor_list = layers.convolution(input_tensor_list, rbf, unit_vectors)
        input_tensor_list = layers.concatenation(input_tensor_list)
        if layer == num_layers - 1:
            with tf.variable_scope(None, 'atom_types', values=[input_tensor_list[0]]):
                atom_type_list = layers.self_interaction({0: input_tensor_list[0]}, num_atom_types)
        input_tensor_list = layers.self_interaction(input_tensor_list, layer_dim)
        if layer < num_layers - 1:
            with tf.variable_scope(None, 'nonlinearity', values=[input_tensor_list]):
                input_tensor_list = layers.nonlinearity(input_tensor_list, nonlin=utils.ssp)

probabilty_scalars = input_tensor_list[0][0]
missing_coordinates = input_tensor_list[1][0]
atom_type_scalars = atom_type_list[0][0]

# [N]
p = tf.nn.softmax(tf.squeeze(probabilty_scalars))

# [N, 3] when layer3_dim == 1
output = tf.squeeze(missing_coordinates)

# votes : [N, 3]
votes = r + output

# guess_coord : [3]
guess_coord = tf.tensordot(p, votes, [[0], [0]])

# guess_atom : [num_atom_types
guess_atom = tf.tensordot(p, tf.squeeze(atom_type_scalars), [[0], [0]])

# missing_point : [3]
missing_point = tf.placeholder(FLOAT_TYPE, shape=(3))
missing_atom_type = tf.placeholder(FLOAT_TYPE, shape=(num_atom_types))

# loss : []
loss = tf.nn.l2_loss(missing_point - guess_coord)
loss += tf.nn.l2_loss(missing_atom_type - guess_atom)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1.e-3
step_learning_rate = 1000
decay_factor = 0.3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           training_set_size*step_learning_rate, decay_factor, staircase=True)

optim = tf.train.AdamOptimizer(learning_rate)

train_op = optim.minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=None)
#saver.restore(sess, "missing_point_checkpoints/qm9_model_200.ckpt")

epochs = 1000
print_freq = 25
save_freq = 25

print("Training model.")

for epoch in range(epochs):
    for shape, types in zip(qm9_coords, qm9_one_hot):
        remove_index = random.randrange(len(shape))
        new_shape = np.delete(shape, remove_index, 0)
        new_types = np.delete(types, remove_index, 0)
        removed_point = shape[remove_index]
        removed_types = types[remove_index]
        loss_to_print, _ = sess.run([loss, train_op], feed_dict={r: new_shape,
                                      missing_point: removed_point,
                                      missing_atom_type: removed_types,
                                      one_hot: new_types})

    if epoch % print_freq == 0:
        loss_sum = 0.
        for shape, types in zip(qm9_coords, qm9_one_hot):
            for remove_index in range(len(shape)):
                new_shape = np.delete(shape, remove_index, 0)
                new_types = np.delete(types, remove_index, 0)
                removed_point = shape[remove_index]
                removed_types = types[remove_index]
                loss_value, guess_point, guess_type, votes_points, probs = sess.run(
                    [loss, guess_coord, guess_atom, votes, p],
                    feed_dict={r: new_shape,
                               missing_point: removed_point,
                               missing_atom_type: removed_types,
                               one_hot: new_types})
                loss_sum += loss_value
        print("train", epoch, np.sqrt(2 * loss_sum / np.sum(list(map(len, qm9_coords)))))

    if epoch % print_freq == 0:
        loss_sum = 0.
        for shape, types in zip(qm9_test_coords, qm9_test_one_hot):
            for remove_index in range(len(shape)):
                new_shape = np.delete(shape, remove_index, 0)
                new_types = np.delete(types, remove_index, 0)
                removed_point = shape[remove_index]
                removed_types = types[remove_index]
                loss_value, guess_point, guess_type, votes_points, probs = sess.run(
                    [loss, guess_coord, guess_atom, votes, p],
                    feed_dict={r: new_shape,
                               missing_point: removed_point,
                               missing_atom_type: removed_types,
                               one_hot: new_types})
                loss_sum += loss_value
        print("test", epoch, np.sqrt(2 * loss_sum / np.sum(list(map(len, qm9_test_coords)))))

    if epoch % save_freq == 0:
        save_path = saver.save(sess, "missing_point_checkpoints/qm9_model_{}.ckpt".format(epoch))

        print("Model saved in path: %s" % save_path)
