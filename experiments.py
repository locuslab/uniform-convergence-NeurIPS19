import setGPU
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
import os, sys
sys.path.append(os.getcwd())
import pprint


flags = tf.app.flags
flags.DEFINE_integer("i_dim", 784, "Input dimensionality [784]")
flags.DEFINE_integer("h_dim", 500, "No. of hidden units in network [1000]")
flags.DEFINE_integer("depth", 5, "No. of hidden layers [5]")
flags.DEFINE_integer("n_train", 4096, "No. of training samples [2000]")
flags.DEFINE_integer("margin", 10, "Margin tolerance [2]")
flags.DEFINE_integer("n_batch", 64, "Batch size [64]")
flags.DEFINE_float("eta", 0.1, "Learning rate [0.1]")
flags.DEFINE_float("threshold", 0.001, "Threshold of training error")
flags.DEFINE_string("tag", " ", "Any details")
flags.DEFINE_boolean("noise", False, "Noise [false]")
flags.DEFINE_string("dataset", "mnist", "Dataset [mnist]")
flags.DEFINE_integer("n_classes", 10, "No. of classes (chosen at random) [5]")

FLAGS = flags.FLAGS


PATH = "~/MNIST_data"
def main(_):
	print(tf.flags.FLAGS.__flags)
	TAG=FLAGS.tag

	# This creates a folder with integer name "idx" for some integer idx that is not already present
	# and stores flag variables in idx/readme.txt, and other outputs of the experiments for the given
	# settings
	idx = 0
	while True:
		if FLAGS.tag != " ":
			TAG = FLAGS.tag+'-'+str(idx)
		else:
			TAG = str(idx)
		if os.path.exists(TAG):
			readme_file = open(TAG+'/readme.txt', 'r')
			readme_txt = readme_file.read().split('\n')
			readme = {}
			for line in readme_txt[:-1]:
				a = line.strip('{},').split(':')   
				a[0] = a[0].strip(' \'') 
				a[1] = a[1].strip(' \'') 
				readme[a[0]] = (a[1])
			readme_file.close()


		if not os.path.exists(TAG):
			os.makedirs(TAG)
			break
		idx += 1

	with open(TAG+"/readme.txt", "w") as out:
		pprint.pprint(tf.flags.FLAGS.__flags,stream=out)



	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(PATH, one_hot=True)


	def unpickle(file):
		import pickle
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict


	def get_random_mnist_train_set(n_train, noise=False):

		#classes = np.random.choice(np.arange(10),size=n_classes,replace=False)
		#select_data = np.sum(mnist.train.labels[:,classes],axis=1)>0


		train_samples_ind = np.random.randint(55000, size=n_train)
		train_samples = mnist.train.images[train_samples_ind,:]

		if noise:
			train_labels = np.zeros((n_train,10)) 
			train_labels_ind = np.random.randint(10,size=n_train)
			train_labels[np.arange(n_train),train_labels_ind]=1.0
		else:
			train_labels = mnist.train.labels[train_samples_ind,:]

		if n_train > 32678:
			train_samples[32678:,] += np.random.uniform(-0.01,0.01, size=train_samples[32678:].shape)
		return train_samples, train_labels

	def get_random_mnist_test_set(n_test):
		test_samples_ind = np.random.permutation(np.arange(mnist.test.labels.shape[0]))[:n_test] # random permutation
		test_samples = mnist.test.images[test_samples_ind,:]
		test_labels = mnist.test.labels[test_samples_ind,:]
		return test_samples, test_labels




	#An auxiliary function to plot stuff, I have not used this in a while.
	def plot_fig(x, xlab, ylab, dest, TAG):
		fig=figure()
		x = np.array(x)
		plot(range(len(x)),x)
		ax = subplot(111)
		ax.set_ylabel(ylab)
		ax.set_xlabel(xlab)
		fig.savefig(TAG+'/' + dest + '.pdf')
		np.savetxt(TAG+"/"+dest+".txt",x)
		close(fig)



	def error(labels, logits):
		''' Computes average 0-1 error '''
		# labels =  a 0-1 matrix of no. of points x 10  values containing the true labels
		# logit = a real-valued matrix of no. of points x 10 values containing the logit predictions
		# Output: average 0-1 error
		label_ind = np.argmax(labels, axis=1)
		predictions = np.argmax(logits,axis=1)
		return np.mean(label_ind != predictions)

	def margin_error(labels, logits, margin):
		''' Computes average margin-based error '''
		# labels =  a 0-1 matrix of no. of points x 10  values containing the true labels
		# logit = a real-valued matrix of no. of points x 10 values containing the logit predictions
		# Output: average no. of points which have not been classified correctly by at least the given margin
		label_ind = np.argmax(labels, axis=1) # correct labels
		modified_logits = np.copy(logits)
		for i in range(logits.shape[0]):
			modified_logits[i,label_ind[i]] = -float("inf")
		max_wrong_logits = np.max(modified_logits,axis=1)
		max_true_logits = logits[[i for i in range(logits.shape[0])],list(label_ind)]
		return np.mean(max_true_logits - max_wrong_logits < margin)

	def margin(labels, logits):
		''' Computes a list of margins  '''
		# labels =  a 0-1 matrix of no. of points x 10  values containing the true labels
		# logit = a real-valued matrix of no. of points x 10 values containing the logit predictions
		# Output: returns a list of size (no. of datapoints) containing the margin for each datapoint
		label_ind = np.argmax(labels, axis=1) # correct labels
		modified_logits = np.copy(logits)
		for i in range(logits.shape[0]):
			modified_logits[i,label_ind[i]] = -float("inf")
		max_wrong_logits = np.max(modified_logits,axis=1)
		max_true_logits = logits[[i for i in range(logits.shape[0])],list(label_ind)]
		return max_true_logits - max_wrong_logits




	# A function to create a TF linear transformation layer
	def linear(input, i_dim, o_dim, scope=None, scale=1.0):
		''' A linear transformation layer '''
		# Code was originally copied from  https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
		# and then modified.
		initializer = tf.contrib.layers.xavier_initializer()
		with tf.variable_scope(scope or 'linear'):
			w = tf.get_variable('w', [input.get_shape()[1], o_dim], initializer=initializer)
			return tf.matmul(input, w)  




	def network(input_var, i_dim, h_dim, o_dim, depth, scale=1.0):
		# we will store the pre/post-activations of each layer 
		# in a dictionary for efficient retreival
		output_dict = {}
		output_dict['pre'] = []
		output_dict['post'] = []
		output_dict['pre'] += [linear(input_var, i_dim, h_dim, 'pre0',scale)]
		output_dict['post'] += [tf.nn.relu(output_dict['pre'][0])]

		for k in range(depth-1):
			# if depth is one, we will not have any  HxH matrix, so we will skip this loop
			# At the entry of a loop, we have a post-activation ready
			output_dict['pre'] += [linear(output_dict['post'][k], h_dim, h_dim, 'pre'+str(k+1),scale)]
			output_dict['post'] += [tf.nn.relu(output_dict['pre'][k+1])]

		#final layer is just linear
		output_dict['pre'] += [linear(output_dict['post'][depth-1], h_dim, o_dim, 'pre'+str(depth),scale)]
		output_dict['output'] = output_dict['pre'][depth] # an alias
		return output_dict




	input_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.n_batch, FLAGS.i_dim))
	label = tf.placeholder(tf.float32, shape=(FLAGS.n_batch, FLAGS.n_classes))



	# Creating the FFN network we will train
	with tf.variable_scope("train"):
		train_output = network(input_placeholder, FLAGS.i_dim, FLAGS.h_dim, FLAGS.n_classes, FLAGS.depth)
	train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "train")

	# For reinitializing the networks
	train_weights_placeholder = [tf.placeholder(dtype=tf.float32, shape=t.shape) for t in train_vars]
	reinitialize_train_op = [train_vars[t].assign(train_weights_placeholder[t]) for t in range(len(train_weights_placeholder))]


	# Training operations for FFN
	cross_entropy_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=train_output['output']))
	opt = tf.train.GradientDescentOptimizer(FLAGS.eta)
	train_op = opt.minimize(cross_entropy_loss_op, var_list=train_vars)
	#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

	sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
	# sess = tf.InteractiveSession()


	def perform_op_over_data(samples, labels, op):
		''' Applies a given operation over all datapoints '''
		# samples: an np array of (no. of points x 784) values
		# labels: an np array of (no. of points x 784) values
		# op: a tensorflow op
		# outputs: this is a list that contains as many elements as the no of batches
		# i.e., (no.of points / batch size). Each element is the output of running op
		# on that batch. So expect this to be a numpy array with batch size many rows.
		outputs = []
		max_batch_ind = int(samples.shape[0]/FLAGS.n_batch)
		for batch_ind in range(max_batch_ind):
			curr_indices = np.array(range(FLAGS.n_batch*batch_ind,FLAGS.n_batch*(batch_ind+1)))
			outputs += [sess.run(op, 
				feed_dict={input_placeholder: samples[curr_indices], 
					label: labels[curr_indices]})]
		return outputs


	def get_error(samples, labels, margin, outputs):
		''' Given labeled data, the margin of the classifier on those data, and the network outputs, returns the margin-based error'''
		error = 0.0
		max_batch_ind = int(samples.shape[0]/FLAGS.n_batch)
		for batch_ind in range(max_batch_ind):
			curr_indices = np.array(range(FLAGS.n_batch*batch_ind,FLAGS.n_batch*(batch_ind+1)))
			error += margin_error(labels[curr_indices], outputs[batch_ind], margin)
		error/=float(max_batch_ind)
		return error


	def get_margin_distribution(samples, labels, outputs):
		''' Given data and the margin of the classifier on those data, and network outputs, returns the list of margins'''
		error = 0.0
		max_batch_ind = int(samples.shape[0]/FLAGS.n_batch)
		margins = np.array([])
		for batch_ind in range(max_batch_ind):
			curr_indices = np.array(range(FLAGS.n_batch*batch_ind,FLAGS.n_batch*(batch_ind+1)))
			margins = np.concatenate([margins,margin(labels[curr_indices], outputs[batch_ind])])
		return margins


	def reinitialize_weights(weights, placeholder, op):
		''' reinitializes weights by assigning the scalar variable weights to the placeholder and running the reinitialization op'''
		feed_dict = {}
		for k in range(len(placeholder)):
			feed_dict[placeholder[k]] = weights[k]
		return sess.run(op, feed_dict = feed_dict)






	sess.run(tf.initialize_all_variables())

	# First entry of initial weights is for the uncompressed network
	initial_weights = sess.run(train_vars)

	# ============================
	# First round of training
	train_samples_1, train_labels_1=  get_random_mnist_train_set(FLAGS.n_train, FLAGS.noise)
	test_samples, test_labels = get_random_mnist_test_set(int(10000/FLAGS.n_batch)*FLAGS.n_batch)
	train_samples_2, train_labels_2 =  get_random_mnist_train_set(FLAGS.n_train, FLAGS.noise)


	epoch=0
	train_margin_error = 1.0
	while train_margin_error > FLAGS.threshold:
		epoch += 1
		# One epoch
		perform_op_over_data(train_samples_1, train_labels_1, train_op)
		outputs = 	perform_op_over_data(train_samples_1, train_labels_1, train_output['output'])
		# Calculate train_margin_error
		train_margin_error = get_error(train_samples_1, train_labels_1, FLAGS.margin, outputs)
		print(train_margin_error)


	outputs = perform_op_over_data(test_samples, test_labels, train_output['output'])
	test_margin_error_1 = get_error(test_samples, test_labels, 0, outputs)
	test_margin_distribution = get_margin_distribution(test_samples, test_labels, outputs)
	np.savetxt(TAG+'/test_margins.txt',test_margin_distribution)



	outputs = 	perform_op_over_data(train_samples_1, train_labels_1, train_output['output'])
	train_margin_error = get_error(train_samples_1, train_labels_1, FLAGS.margin, outputs)
	np.savetxt(TAG+'/margins.txt',get_margin_distribution(train_samples_1, train_labels_1, outputs))





	# Store weights learned by training on first random draw 
	final_weights_1 = sess.run(train_vars)


	# ============================
	# Second round of training
	# Reinitialize network with same random initialization as before
	_ = reinitialize_weights(initial_weights, train_weights_placeholder, reinitialize_train_op)


	epoch=0
	train_margin_error = 1.0
	while train_margin_error > FLAGS.threshold:
		epoch += 1
		# One epoch
		perform_op_over_data(train_samples_2, train_labels_2, train_op)
		outputs = 	perform_op_over_data(train_samples_2, train_labels_2, train_output['output'])
		# Calculate train_margin_error
		train_margin_error = get_error(train_samples_2, train_labels_2, FLAGS.margin, outputs)
		print(train_margin_error)
	outputs = perform_op_over_data(test_samples, test_labels, train_output['output'])
	test_margin_error = get_error(test_samples, test_labels, 0, outputs)
	final_weights_2 =  sess.run(train_vars)

	outputs = perform_op_over_data(test_samples, test_labels, train_output['output'])
	test_margin_error_2 = get_error(test_samples, test_labels, 0, outputs)

	np.savetxt(TAG+"/test_errors.txt",[test_margin_error_1,test_margin_error_2])



	# Save values

	distance_moved_1 = [np.linalg.norm(weight_1-weight_2) for (weight_1, weight_2) in zip(final_weights_1,initial_weights)]
	np.savetxt(TAG+"/distance_from_initialization.txt",distance_moved_1)

	distance_moved_2 = [np.linalg.norm(weight_1-weight_2) for (weight_1, weight_2) in zip(final_weights_1,final_weights_2)]
	np.savetxt(TAG+"/distance_between_weights.txt",distance_moved_2)

	spectral_norms = [np.linalg.norm(np.transpose(weight),ord=2) for weight in final_weights_1]
	np.savetxt(TAG+"/spectral_norm.txt",spectral_norms)

	frobenius_norms = [np.linalg.norm(np.transpose(weight)) for weight in final_weights_1]
	np.savetxt(TAG+"/frobenius_norm.txt",frobenius_norms)

	# NOTE: All these compute only the NUMERATOR of the generalization bounds.
	# We divide these by the denominator before plotting in the paper.
	# Neyshabur+ ICLR'17
	pac_spectral_bound_0 = np.max(np.linalg.norm(train_samples_1, axis=1))*FLAGS.depth*np.sqrt(FLAGS.h_dim)
	# pac_spectral_bound *= np.log(FLAGS.depth*FLAGS.h_dim)
	pac_spectral_bound_0 *= np.linalg.norm([np.linalg.norm(final_weights_1[k])/spectral_norms[k] for k in range(len(final_weights_1))])
	pac_spectral_bound_0 *= np.prod(spectral_norms)


	# Neyshabur+ ICLR'17 with distance from initialization (plotted on our paper)
	pac_spectral_bound_1 = np.max(np.linalg.norm(train_samples_1, axis=1))*FLAGS.depth*np.sqrt(FLAGS.h_dim)
	# pac_spectral_bound *= np.log(FLAGS.depth*FLAGS.h_dim)
	pac_spectral_bound_1 *= np.linalg.norm([np.linalg.norm(final_weights_1[k]-initial_weights[k])/spectral_norms[k] for k in range(len(final_weights_1))])
	pac_spectral_bound_1 *= np.prod(spectral_norms)

	# Neyshabur+ ICLR'17 with distance between weights
	pac_spectral_bound_2 = np.max(np.linalg.norm(train_samples_1, axis=1))*FLAGS.depth*np.sqrt(FLAGS.h_dim)
	# pac_spectral_bound *= np.log(FLAGS.depth*FLAGS.h_dim)
	pac_spectral_bound_2 *= np.linalg.norm([np.linalg.norm(final_weights_1[k]-final_weights_2[k])/spectral_norms[k] for k in range(len(final_weights_1))])
	pac_spectral_bound_2 *= np.prod(spectral_norms)


	# Bartlett+'17 bound
	covering_spectral_bound_1 = np.max(np.linalg.norm(train_samples_1, axis=1))
	covering_spectral_bound_1 *= np.power(np.sum([ np.power(np.sum(np.linalg.norm(final_weights_1[k]-initial_weights[k],axis=1))/spectral_norms[k],2.0/3.0) for k in range(len(final_weights_1))]),1.5)
	covering_spectral_bound_1 *= np.prod(spectral_norms)


	covering_spectral_bound_2 = np.max(np.linalg.norm(train_samples_1, axis=1))
	covering_spectral_bound_2 *= np.power(np.sum([ np.power(np.sum(np.linalg.norm(final_weights_1[k]-final_weights_2[k],axis=1))/spectral_norms[k],2.0/3.0) for k in range(len(final_weights_1))]),1.5)
	covering_spectral_bound_2 *= np.prod(spectral_norms)

	# Neyshabur et al '19, ignoring the sqrt h
	# This applies to only 1 hidden layer networks, i.e., depth =1
	unilayer_bound = np.linalg.norm(initial_weights[0],ord=2)*np.linalg.norm(final_weights_1[1])
	unilayer_bound += np.linalg.norm(final_weights_1[0]-initial_weights[0])*np.linalg.norm(final_weights_1[1])
	unilayer_bound *= np.max(np.linalg.norm(train_samples_1, axis=1))


	np.savetxt(TAG+"/bounds.txt",[pac_spectral_bound_0,pac_spectral_bound_1,pac_spectral_bound_2,covering_spectral_bound_1,covering_spectral_bound_2, unilayer_bound])




if __name__ == '__main__':
  tf.app.run()



