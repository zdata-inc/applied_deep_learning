import os
import string
#import argparse

import numpy as np
import tensorflow as tf

#parser = argparse.ArgumentParser(description='Kaggle DSB2018 Training')
#parser.add_argument('--phrase', default='piglatin', type=str, 
#                    help='which english phrase to translate into pig latin')
#args = parser.parse_args()

eng_phrase = ''

chars = string.ascii_letters + '>_'
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

def load_graph(graph_flnm):
    with tf.gfile.GFile(graph_flnm, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        
        # the prefix will be import by default, so we'll give it something meaningful
        tf.import_graph_def(graph_def, name='enc-dec')

    return graph
    
graph = load_graph('./saved_translator/piglatin_enc-dec.pb')

with tf.Session(graph=graph) as sess:
    word_input = graph.get_tensor_by_name('enc-dec/encoder_input:0')
    predictions = graph.get_tensor_by_name('enc-dec/decoder_pred:0')

    # clear the terminal after all the tensorflow stuff loads
    os.system('clear')

    while eng_phrase != 'Q':
        eng_phrase = input('Enter a phrase to translate: ')

        if eng_phrase == 'Q':
            break
    
        output_translation = []
        for word in eng_phrase.split():
        
            val_input = [char2idx[c] for c in word]
            val_input = np.asarray(val_input).reshape(1, len(val_input))

            # for the inference mode we only pass the english word to translate
            prediction = sess.run(predictions, feed_dict={word_input: val_input})

             
            output_translation.append(''.join([idx2char[idx] for idx in prediction[0]]))
            
        print(' '.join(output_translation).replace('_', ''))