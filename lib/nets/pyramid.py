# ------------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zac Wellmer 
# with influence from Zheqi He, Xinlei Chen, and Charles Shang
# ------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
import numpy as np

from nets.network import Network
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.snippets import generate_anchors_pre
from layer_utils.wrapper import assign_boxes
from model.config import cfg

_networks_map = {
    'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
                 'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
                 'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
                 'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
                 'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
                },
}

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.RESNET.BN_TRAIN,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

def my_sigmoid(x):
    """add an active function for the box output layer, which is linear around 0"""
    return (tf.nn.sigmoid(x) - tf.cast(0.5, tf.float32)) * 6.0

class Pyramid(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._predictions = {i: {} for i in range(5, 1, -1)}
    self._anchor_targets = {i: {} for i in range(5, 1, -1)}
    self._proposal_targets = {i: {} for i in range(5, 1, -1)}
    self._num_layers = num_layers
    self._arch = 'res_v1_%d' % num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers
    
    self._feat_stride = {}

  def _anchor_component(self, p_i, h, w):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [h, w,
                                           self._feat_stride[p_i], self._anchor_scales[p_i], self._anchor_ratios],
                                           [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length   

  def _crop_pool_layer(self, bottom, rois, name, p_i):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[p_i][0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[p_i][0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.RESNET.MAX_POOL:
        pre_pool_size = cfg.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE],
                                         name="crops")
    return crops

  def _anchor_target_layer(self, rpn_cls_score, name, p_i):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride[p_i], self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32])
      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])
      
      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets[p_i]['rpn_labels'] = rpn_labels
      self._anchor_targets[p_i]['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets[p_i]['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets[p_i]['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
      
      #self._score_summaries.update(self._anchor_targets) #  score summaries not compatible with dict
    return rpn_labels

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name, p_i):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride[p_i], self._anchors, self._num_anchors],
                                     [tf.float32, tf.float32]) 
  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name, p_i):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride[p_i], self._anchors, self._num_anchors],
                                     [tf.float32, tf.float32])
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])
    return rois, rpn_scores

  def _proposal_target_layer(self, rois, roi_scores, name, p_i):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets[p_i]['rois'] = rois
      self._proposal_targets[p_i]['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets[p_i]['bbox_targets'] = bbox_targets
      self._proposal_targets[p_i]['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets[p_i]['bbox_outside_weights'] = bbox_outside_weights

      return rois, roi_scores

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
    return net

  def build_pyramid(self, endpoints, bilinear=True):
    pyramid = {}
    pyramid_map = _networks_map['resnet50']
    with tf.name_scope('pyramid'):
      pyramid[5] = slim.conv2d(endpoints[pyramid_map['c5']], 256, [1, 1], stride=1, scope = 'c5')
      for c in range(4, 1, -1):
        s, s_ = pyramid[c+1], endpoints[pyramid_map['c%d'%c]]
        up_shape = tf.shape(s_)
        s = tf.image.resize_bilinear(s, [up_shape[1], up_shape[2]], name='c%d/upscale'%c)
        s_ = slim.conv2d(s_, 256, [1,1], stride=1, scope='c%d'%c)
        s = tf.add(s, s_, name='c%d/addition'%c)
        s = slim.conv2d(s, 256, [3,3], stride=1, scope='c%d/fusion'%c)
        pyramid[c] = s     
    return pyramid

  def build_network(self, sess, is_training=True):
    #  pyramid network scales changes at different levels of pyramid
    self._anchor_scales = {}
 
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 101:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 152:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError
    assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 4)
    if cfg.RESNET.FIXED_BLOCKS == 4:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, endpoints = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    elif cfg.RESNET.FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, endpoints = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, endpoints = resnet_v1.resnet_v1(net,
                                           blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, endpoints = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    pyramid = self.build_pyramid(endpoints)
    self._layers['head'] = net_conv4  # not sure what to do with this
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      for i in range(5, 1, -1):
        p = i
        self._act_summaries.append(pyramid[p])
        self._feat_stride[p] = [2 ** i]
        shape = tf.shape(pyramid[p])
        h, w = shape[1], shape[2]
        
        #  in the paper they use only one anchor per layer of pyramid. But when I tried that we were frequently receiving no overlaps in anchor_target_proposal(...) 
        self._anchor_scales[p] = [2**(i-j) for j in range(self._num_scales-1, -1, -1)]
        self._anchor_component(p, h, w)

        # build the anchors for the image
        # rpn
        rpn = slim.conv2d(pyramid[p], 256, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3", stride=1)
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      
        if is_training:
          rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", p)
          rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor", p)
          # Try to have a determinestic order for the computing graph, for reproducibility
          with tf.control_dependencies([rpn_labels]):
            rois, roi_scores = self._proposal_target_layer(rois, roi_scores, "rpn_rois", p)
        else:
          if cfg.TEST.MODE == 'nms':
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois", p)
          elif cfg.TEST.MODE == 'top':
            rois, roi_scores = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois", p)
          else:
            raise NotImplementedError
        self._predictions[p]['rois'] = rois
        self._predictions[p]['roi_scores'] = roi_scores
        self._predictions[p]['rpn_cls_score'] = rpn_cls_score 
        self._predictions[p]['rpn_cls_score_reshape'] = rpn_cls_score_reshape
        self._predictions[p]['rpn_cls_prob'] = rpn_cls_prob
        self._predictions[p]['rpn_bbox_pred'] = rpn_bbox_pred
    
    all_roi_scores = tf.concat(values=[self._predictions[p]['roi_scores'] for p in pyramid], axis=0)
    all_rois = tf.concat(values=[self._predictions[p]['rois'] for p in pyramid], axis=0)
    p_vals = [tf.fill([tf.shape(self._predictions[p]['roi_scores'])[0], 1], float(p)) for p in pyramid]
    p_roi = tf.concat(values=[tf.reshape(p_vals, [-1, 1]), all_rois], axis=1)
    
    if is_training:
      all_proposal_target_labels = tf.concat(values=[self._proposal_targets[p]['labels'] for p in pyramid], axis=0)
      all_proposal_target_bbox = tf.concat(values=[self._proposal_targets[p]['bbox_targets'] for p in pyramid], axis=0)
      all_proposal_target_inside_w = tf.concat(values=[self._proposal_targets[p]['bbox_inside_weights'] for p in pyramid], axis=0)
      all_proposal_target_outside_w = tf.concat(values=[self._proposal_targets[p]['bbox_outside_weights'] for p in pyramid], axis=0)

    cfg_key = self._mode
    if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
    nms_top_n = all_roi_scores.shape[0] \
                    if all_roi_scores.shape[0] < cfg[cfg_key].RPN_POST_NMS_TOP_N \
                    else cfg[cfg_key].RPN_POST_NMS_TOP_N
    _, top_indices = tf.nn.top_k(tf.reshape(all_roi_scores, [-1]), k=nms_top_n)
    p_roi = tf.gather(p_roi, top_indices)
    
    [assigned_rois, _, _] = \
                assign_boxes(all_rois, [all_rois, top_indices], [2, 3, 4, 5], 'assign_boxes')

    for p in range(5, 1, -1):
      splitted_rois = assigned_rois[p-2]

      # rcnn 
      if cfg.POOLING_MODE == 'crop':
        cropped_roi = self._crop_pool_layer(pyramid[p], splitted_rois, "cropped_roi", p) 
        self._predictions[p]['cropped_roi'] = cropped_roi
      else:
        raise NotImplementedError
    cropped_rois = [self._predictions[p_layer]['cropped_roi'] for p_layer in self._predictions]
    cropped_rois = tf.concat(values=cropped_rois, axis=0)


    cropped_regions = slim.max_pool2d(cropped_rois, [3, 3], stride=2, padding='SAME')
    refine = slim.flatten(cropped_regions)
    refine = slim.fully_connected(refine, 1024, activation_fn=tf.nn.relu)
    refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
    refine = slim.fully_connected(refine,  1024, activation_fn=tf.nn.relu)
    refine = slim.dropout(refine, keep_prob=0.75, is_training=is_training)
    cls_score = slim.fully_connected(refine, self._num_classes, activation_fn=None, 
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    bbox_pred = slim.fully_connected(refine, self._num_classes*4, activation_fn=my_sigmoid, 
            weights_initializer=tf.truncated_normal_initializer(stddev=0.001))

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = tf.gather(all_rois, top_indices)
    
    if is_training:
      self._proposal_targets['labels'] = all_proposal_target_labels 
      self._proposal_targets['bbox_targets'] = all_proposal_target_bbox
      self._proposal_targets['bbox_inside_weights'] = all_proposal_target_inside_w
      self._proposal_targets['bbox_outside_weights'] = all_proposal_target_outside_w

    #self._score_summaries.update(self._predictions) #  score summaries not compatible w/ dict
    return self._predictions["rois"], cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)
    
    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)
        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))

  def _add_losses(self, sigma_rpn=3.0):
    rpn_box_losses, rpn_cls_losses = [], []
    with tf.variable_scope('loss_' + self._tag) as scope:
      for p in range(5, 1, -1):
        # RPN, class loss
        rpn_cls_score = tf.reshape(self._predictions[p]['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(self._anchor_targets[p]['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions[p]['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets[p]['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets[p]['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets[p]['rpn_bbox_outside_weights']

        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])

      cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      self._losses['total_loss'] = loss

      self._event_summaries.update(self._losses)

    return loss 
