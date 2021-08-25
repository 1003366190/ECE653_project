from cleverhans.tf2.utils import compute_gradient
import numpy as np
import tensorflow as tf

# the fuzz step of pixel fuzz
# requires x to be 1*width*height*3 dimension np array reprresentation of the image 
def pixel_fuzz(model, x, epsilon, num_pixels=10):
    # compute the gradient of each rgb values with respect to the loss function
    y = tf.argmax(model(x), 1)
    grad = compute_gradient(
        model, tf.nn.sparse_softmax_cross_entropy_with_logits, x, y, False)
    # find the L2 norm of the 3 rgb values of each pixel, flatten the array for convinience in computing
    sum_flatten = np.reshape(tf.norm(grad, ord=1, axis=3), -1)
    # flatten the input
    x_flatten = np.array(np.reshape(x, (x.shape[1]*x.shape[2], 3)))
    # flatten the gradient
    grad_flatten = np.reshape(grad, (x.shape[1]*x.shape[2], 3))
    # find the index of the top "num_pixels" pixels with larges gradient L2norm in the flatten array 
    indices = np.argpartition(sum_flatten, -1*num_pixels)[-1*num_pixels:]
    # get the signs of the gradient for the top num_pixel pixels
    signs = np.sign(grad_flatten[indices])
    # generate the output that favours teh direction of gradient in range (-0.5 epsilon , epsilon) 
    updates = tf.random.uniform(signs.shape, -0.5,1)*signs*epsilon
    # update the images selected pixel 
    x_flatten[indices] = x_flatten[indices] + updates
    # return the image to original shape and clip the image to range (0,1)
    output = np.clip(x_flatten, 0, 1)
    output = tf.reshape(output, x.shape)
    return output
