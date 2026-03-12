import tensorflow as tf
def senet_layer(inputs,reduction_ratio=6,seed=1024,name='senet'):
    """

    :param inputs: list of tensors [B,E],长度为F field个数
    :param reduction_ratio: 压缩比例为6
    :param seed:
    :param name:
    :return: [B,F,D]
    """
    stacked_inputs=tf.stack(inputs,axis=1) #[B,F,D]
    field_num=stacked_inputs.get_shape().as_list()[1]
    embed_dim=stacked_inputs.get_shape().as_list()[2]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        mean_inputs=tf.reduce_mean(stacked_inputs,axis=-1) #[B,F]
        reduced_size=int(max(1.0,field_num//reduction_ratio))
        a1=tf.layers.dense(inputs=mean_inputs,
                          units=reduced_size,
                          kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                          activation=tf.nn.relu,name="w1_reduce")
        #[B,R]
        weights=tf.layers.dense(inputs=a1,
                               units=field_num,
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                               name="w2_expand")
        #[B,F]
        weights_reshape=tf.expand_dims(weights,axis=-1) #[B,F,1]
        output=stacked_inputs*weights_reshape #[B,F,D]
        #return1:return output
        senet_result=tf.concat([stacked_inputs,output],axis=2) #[B,F,2*D]
        senet_result=tf.reshape(senet_result,[-1,2*field_num*embed_dim]) #[B,2*F*D]
        return senet_result