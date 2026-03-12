import tensorflow as tf
def din_attention(query,facts,mask,hidden_units,name="din"):
    """
    :param query: [B,E]
    :param facts: [B,T,E]
    :param mask: [B,T]
    :param name:
    :return:[B,E]
    """
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        query=tf.expand_dims(query,axis=1) #[B,E]>[B,1,E]
        seq_len=tf.shape(facts)[1]
        query_tiled=tf.tile(query,[1,seq_len,1]) #[B,1,E]>[B,T,E]
        din_input=tf.concat([query_tiled,facts,query_tiled-facts,query_tiled*facts],axis=-1) #[B,T,4E]
        net=din_input
        for i,unit in enumerate(hidden_units):
            net=tf.layers.dense(inputs=net,units=unit,activation=tf.nn.relu,name='f{}_din'.format(i))
        output_layer = tf.layers.dense(inputs=net,1, activation=None, name='f_out_din') #[B,T,1]
        scores=tf.reshape(output_layer,[-1,1,seq_len]) #[B,1,T]
        if not mask:
            mask_float=tf.cast(mask,tf.float32) #[B,T]
            mask_t=tf.expand_dims(mask_float,axis=1) #[B,1,T]
            scores=scores*mask_t
        din_output=tf.matmul(scores,facts) #[B,1,E]
        din_output=tf.squeeze(din_output,axis=1) #[B,E]
        return din_output

