# decision tree Classifier :deciduous_tree:

This repo serves as a full implementation of a decision tree in Python using numpy, pandas, copy, scipy, pprint and tqdm.

This implementation contains a few main function:

1. build_tree : Creating a decision tree from a given csv file - the creation is done by calculating the overall_entropy at each node of the tree. We will split the data over and over til we reach a leaf (classification).

2. chai_test : After the tree creation we'll make sure that each split has a statistical significance by performing Chi-square pruning :scissors::leaves: - the result should be a smaller tree , less prone to overfitting, a more generalised tree if you will.

3.tree_error (K-fold Cross-Validation) : responsible of saving all error values (after validating the k fold) and print the average when process is done 

4.classify_example : a method that tries to *predict* the calssification of record, that is not a part of the train or testing data, using a decision tree.

Assignment instructions can be found in ex3.pdf