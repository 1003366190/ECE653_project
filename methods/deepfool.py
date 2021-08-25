import copy
import numpy as np
import tensorflow as tf

# Code source https://github.com/MyRespect/AdversarialAttack/blob/master/deepfool_tf2/deepfool_tf.py

def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50, shape=(32, 32, 3)):
    image_array = np.array(image)
    image_norm = tf.cast(image_array, tf.float32)
    image_norm = np.reshape(image_norm, shape) 
    image_norm = image_norm[tf.newaxis, ...]  

    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = label

    def loss_func(logits, I, k):
        return logits[0, I[k]]

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            loss_value = loss_func(fs, I, 0)
        grad_orig = tape.gradient(loss_value, x)

        for k in range(1, num_classes):
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                loss_value = loss_func(fs, I, k)
            cur_grad = tape.gradient(loss_value, x)

            w_k = cur_grad - grad_orig

            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, -1))

            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image_norm + (1 + overshoot) * r_tot

        x = tf.Variable(pert_image)

        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image