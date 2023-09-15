from keras import backend as K
import tensorflow as tf

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, postprocessors=None):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.postprocessors = postprocessors
    
    def decode(self, y_pred, input_length):
        if y_pred.ndim==2:
            y_pred=tf.expand_dims(y_pred,0)
            input_length=[75]
        decoded= tf.keras.backend.ctc_decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths)[0][0].numpy()
        
        #print('decoded:',len(decoded),decoded)
        preprocessed = []
        for output in decoded:
            out = output
            #print(out)
            for postprocessor in self.postprocessors:
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed