path = "/home/tomerweiss/MIMO/summary/cont_lin2/7/" \
       "big_learn_random_1e-3_long/events.out.tfevents.1621356681.floria.59697.0"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from matplotlib.pyplot import imshow, show

image_str = tf.placeholder(tf.string)
im_tf = tf.image.decode_image(image_str)

sess = tf.InteractiveSession()
with sess.as_default():
    count = 0
    for e in tf.train.summary_iterator(path):
        if e.step == 1790:
            for v in e.summary.value:
                if v.tag == 'Rx_selection':
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    imshow(im)
                    show()
