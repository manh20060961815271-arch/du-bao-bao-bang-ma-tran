import tensorflow.compat.v1 as tf

def add_layer(inputs, in_shape, out_shape, activation_function=None, use_layernorm=False, layer_name="m_layer"):
    n1, n2 = in_shape[1], in_shape[2]
    m1, m2 = out_shape[1], out_shape[2]

    # Khởi tạo độ lệch chuẩn 
    init_std = (n1 * n2) ** -0.25
    
 
    with tf.variable_scope(layer_name):
        W = tf.Variable(tf.random.normal([m1, m2, n1], stddev=init_std), name="W_L")   
        V = tf.Variable(tf.random.normal([m1, m2, n2], stddev=init_std), name="W_R")   
        biases = tf.Variable(tf.zeros([1, m1, m2]), name="theta")                      

        # Lan truyền tiến 
        u = tf.einsum('bpq,jkp,jkq->bjk', inputs, W, V) + biases

        if use_layernorm:
            mean, variance = tf.nn.moments(u, axes=[1, 2], keepdims=True)
            gamma = tf.Variable(tf.ones([1, m1, m2]), name="gamma")
            beta = tf.Variable(tf.zeros([1, m1, m2]), name="beta")
            u = gamma * ((u - mean) / tf.sqrt(variance + 1e-6)) + beta

        outputs = u if activation_function is None else activation_function(u)
        
    return W, V, outputs