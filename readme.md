################################
What does the code do.
################################
The file experiments.py contains all the code required to run our experiments. Each run of this script trains a fully connected feedforward neural network on two random draws of a subset of the MNIST dataset from the same initialization. The results of the experiment, (such as the numerical values of the NUMERATOR of the generalization bounds) are saved inside a new folder. The folder name is automatically assigned to be the lowest available integer in the parent folder.

Please see spec-file.txt and package-list.txt for the dependencies.

The file uniform-convergence.ipynb is Jupyter Notebook that presents the code in a much more aesthetic format (although for slightly different parameter settings). This notebook is used for our blog that can be found at https://locuslab.github.io/2019-07-09-uniform-convergence/

The rest of the document describes experiments.py


################################ 
How to run the code.
################################
The script has six options relevant to the paper that can be used as follows:

python experiments.py --h_dim=1024 --depth=5 --n_train=4096  --margin=10 --n_batch=64 --threshold=0.01
 

h_dim sets the width of the network.
depth corresponds to the number of hidden layers in the network.
n_train is the size of the training dataset.
margin and threshold determine the number of epochs until which the network is optimized to minimize the cross entropy loss. Specifically, the code will stop at the epoch when at least 1-threshold of the training data is classified by the given margin.


################################
Where do I find the results.
################################


The results of the experiments are stored as follows:

readme.txt: contains the hyperparameters of the experiment.

bounds.txt: 	
Contains a list of 6 numerical values corresponding to the NUMERATOR of the following generalization bounds (in the paper, we plot this divided by the denominator):
The first value is the PAC-Bayes-based bound from Neyshabur et al '18
The second value is the same bound replaced with distance from initialization.
The third value is the same bound replaced with distance from weights learned on another subset.
The fourth value is the covering number based bound from Bartlett et al '17
The fifth value is the same bound but with distance from initialization (not l2).
The final value is the bound from Neyshabur'19 et al., that applies only to single hidden layer networks.

distance_between_weights.txt:
Contains a list of values corresponding to the l2 norm between the weights learned on two different datasets from the same initialization. Each value corresponds to the weight matrix at a particular depth (with the first value corresponding to the weight matrix following the input layer).

distance_from_initialization.txt:
Contains l2 norms of the update matrices.

spectral_norm.txt:
Contains spectral norms of the learned weight matrices.

frobenius_norm.txt:
Contains the frobenius norms of the learned weight matrices.

test_errors.txt:
Contains the two test errors from the two runs.


margins.txt:
Contains the margin of the network on each training datapoint.


test_margins.txt:
Contains the margin of the network on each test datapoint.





