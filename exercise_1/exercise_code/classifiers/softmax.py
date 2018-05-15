"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = y.shape[0]
    D, C = W.shape
    probs = np.zeros(C)

    for n in range(N):
        # Softmax function
        for c in range(C):
            probs[c] = np.exp(np.dot(X[n], W[:, c]))
        probs /= np.sum(probs)

        # Cross entropy loss for this sample
        loss -= np.log(probs[y[n]])

        # Calculate gradient for this sample
        probs[y[n]] -= 1
        for d in range(D):
            dW[d] += (X[n, d] * probs)

    # Average of all samples
    loss /= N
    dW /= N

    # Regularization (L2?)
    loss += reg * np.sum(W**2)

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = y.shape[0]
    D, C = W.shape
    probs = np.exp(np.dot(X, W))
    probs /= np.sum(probs, axis=1).reshape(-1, 1)

    mask = np.tile(range(C), (N, 1))
    mask = np.equal(mask, y.reshape(-1, 1))

    loss = np.mean(np.log(probs[mask])) * (-1) + reg * np.sum(W**2)
    dW = np.dot(np.transpose(X), probs - mask) / N

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [10 ** i for i in np.linspace(-5.3, -5.1, num=10)]
    regularization_strengths = [10 ** i for i in np.linspace(4, 5, num=20)]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    total = len(learning_rates) * len(regularization_strengths)
    iter = 0

    for l in learning_rates:
        for r in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=l, reg=r, num_iters=1500)
            
            y_train_pred = softmax.predict(X_train)
            y_train_acc = np.mean(y_train == y_train_pred)

            y_val_pred = softmax.predict(X_val)
            y_val_acc = np.mean(y_val == y_val_pred)

            results[l, r] = (y_train_acc, y_val_acc)
            if y_val_acc > best_val:
                best_val = y_val_acc
                l_opt = l
                r_opt = r

            iter += 1
            print("%d / %d" % (iter, total))

    best_softmax = SoftmaxClassifier()
    best_softmax.train(X_train, y_train, learning_rate=l_opt, reg=r_opt, num_iters=4000)

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
