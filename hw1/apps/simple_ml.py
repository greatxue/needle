"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # Pasted from hw0, where more details could be found.
    with gzip.open(image_filename, 'rb') as img_file:
        magic_num, img_num, row, col = struct.unpack('>iiii', img_file.read(16)) 
        img_size = row * col
        
        imgs = [np.array(struct.unpack(f"{img_size}B", img_file.read(img_size)), dtype=np.float32) 
                for _ in range(img_num)] 
        imgs = np.vstack(imgs) 
        X = (imgs - np.min(imgs)) / np.max(imgs) 
    
    with gzip.open(label_filename, "rb") as label_file:
        magic_num, label_num = struct.unpack(">ii", label_file.read(8)) 
        y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8) 

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # A one-hot y as tensors are compatible with direct matrix multiplication.
    # A tuples is applied for axes=(1,) instead of axes=1, as without optimizationit requires an 
    #  iterable object here.
    lhs = ndl.log(ndl.exp(Z).sum(axes=(1,))).sum() # sum to (B, 1) and then scalar
    rhs = (y_one_hot * Z).sum()                    # EW (B, k) and then scalar
    
    return (lhs - rhs) / Z.shape[0]                # divided by batch size 
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_batches = (num_examples + batch - 1) // batch 
    
    for i in range(num_batches):
        start = i * batch
        end = start + batch 
        
        X_batch = ndl.Tensor(X[start:end, :])
        y_batch = y[start:end]
        
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        Z = ndl.relu(X_batch.matmul(W1)).matmul(W2)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
