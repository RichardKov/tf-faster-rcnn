# -----------------------------------------------------------
# Written by CharlesShang@github for Mask-RCNN implementation
# Modified by Zac Wellmer
# -----------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import assign

def assign_boxes(gt_boxes, tensors, layers, scope):
    with tf.name_scope(scope) as sc:
        min_k = layers[0]
        max_k = layers[-1]
        assigned_layers = \
            tf.py_func(assign.assign_boxes, 
                     [ gt_boxes, min_k, max_k ],
                     tf.int32)
        assigned_layers = tf.reshape(assigned_layers, [-1])

        assigned_tensors = []
        for t in tensors:
            split_tensors = []
            for l in layers:
                tf.cast(l, tf.int32)
                inds = tf.where(tf.equal(assigned_layers, l))
                inds = tf.reshape(inds, [-1])
                split_tensors.append(tf.gather(t, inds))
            assigned_tensors.append(split_tensors)

        return assigned_tensors + [assigned_layers]

