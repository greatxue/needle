#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

/* Python implementation here for reference:
 *
 *  num_examples = X.shape[0]
 *  num_batches = (num_examples + batch - 1) // batch 
 *  for i in range(num_batches):
 *      start = i * batch
 *      end = start + batch
 *      X_batch = X[start:end]
 *      y_batch = y[start:end]
 *
 *      exp_logits = np.exp(X_batch @ theta) 
 *      grad = exp_logits / np.sum(exp_logits, axis=1, keepdims = True) 
 *      grad[np.arange(batch), y_batch] -= 1 
 *       
 *      d_theta =  X_batch.T @ (grad / batch)
 *      theta -= lr * d_theta
 */  


/* Implement a matrix multiplication function for further operations.   
 * Initialize an empty Z, and then execute X @ Y = Z in-place.   
 * In detail, \sigma x_is * y_sj = z_ij, (m, n) * (n, k) = (m, k)
 */
void mat_mul(const float * X, const float * Y, float * Z, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            Z[i * k + j] = 0; // first initialize the elements in Z to be 0s
            for (int s = 0; s < n; s++) {
                Z[i * k + j] += X[i * n + s] * Y[s * k + j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    /* Always keep in mind the shapes -> X: (m, n), y: (1, m), theta: (n, k). */ 
    int batch_num = (m + batch - 1) / batch;
    
    for (int iter = 0; iter < batch_num; iter++) {
        /* As the matrix represents as arrays in C++, it will point to the start of each iteration. */
        const float *X_batch = &X[iter * batch * n];
        
        /* To compute exp_logits, first initialize an array of shape (B, k) = (B, n) * (n, k). */
        float *exp_logits = new float[batch * k];
        mat_mul(X_batch, theta, exp_logits, batch, n, k); // logits here actually
        /* Exponential the (B, k) exp_logits one-by-one, in the from of array. */
        for (size_t i = 0; i < batch * k; i++) exp_logits[i] = exp(exp_logits[i]); 
        
        /* Sum up the matrix in axis=1, i.e. in the dimension of 'k'. */
        for (size_t i = 0; i < batch; i++) {
            float sum = 0;
            for (size_t j = 0; j < k; j++) sum += exp_logits[i * k + j];
            for (size_t j = 0; j < k; j++) exp_logits[i * k + j] /= sum; // grad here actually
        }
        
        /* The exp_logits is updated always in an iteration, first we need to locate the specific 
         *  column i*k, and select the corresponding classification, which is that in y. But we need 
         *  to add iter*batch beforehand for y to initialize at the start as well.
         */
        for (size_t i = 0; i < batch; i++) exp_logits[i * k + y[iter * batch + i]] -= 1;
        
        /* Perform a transpose of X_batch, i.e. x_ji -> x_ij. */
        float *X_batch_T = new float[n * batch];
        for (size_t i = 0; i < batch; i++) {
            for (size_t j = 0; j < n; j++) {
                X_batch_T[j * batch + i] = X_batch[i * n + j];
            }
        }
         
        float *d_theta = new float[n * k];
        mat_mul(X_batch_T,exp_logits, d_theta, n, batch, k);
        
        for (size_t i = 0; i < n * k; i++) theta[i] -= lr / batch * d_theta[i];
        
        /* Don't forget to release the memory for eaxh batch iteration. */
        delete[] exp_logits;
        delete[] X_batch_T;
        delete[] d_theta;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
