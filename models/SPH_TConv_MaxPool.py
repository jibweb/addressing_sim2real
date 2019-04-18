import numpy as np
import tensorflow as tf
from utils.logger import TimeScope
from utils.tf import fc, fc_bn, define_scope
from utils.params import params as p

from layers import conv2d_bn, point_conv

MODEL_NAME = "TConv_MaxPool"

# Dropout prob params
# p.define("pool_drop_prob", 0.5)
# Model arch params
# p.define("residual", False)
p.define("tconv_hid_units", [16])
# p.define("red_hid_units", [256, 64])


class Model(object):
    def __init__(self,
                 bn_decay=None):
        # --- I/O Tensors -----------------------------------------------------
        with TimeScope(MODEL_NAME + "/placeholder_setup", debug_only=True):
            self.bias_mat = tf.placeholder(tf.float32,
                                           (None,
                                            p.nodes_nb,
                                            p.nodes_nb),
                                           name="bias_mat")

            self.tconv_index = tf.placeholder(tf.int32,
                                              (None,
                                               9),
                                              name="tconv_index")

            self.valid_pts = tf.placeholder(tf.float32,
                                            [None,
                                             p.nodes_nb,
                                             p.nodes_nb],
                                            name="valid_pts")

            self.num_channels = sum(p.feat_config.values())
            if p.feat_config["lscm"]:
                self.num_channels -= 1

            self.node_feats = tf.placeholder(tf.float32,
                                             (None,
                                              p.nodes_nb,
                                              p.feat_nb[0],  # image size
                                              p.feat_nb[0],  # image size
                                              self.num_channels),
                                             name="node_feats")

            self.y = tf.placeholder(tf.float32,
                                    [None, p.num_classes],
                                    name="y")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.pool_drop = tf.placeholder(tf.float32, name="pool_drop_prob")
            self.bn_decay = bn_decay

            self.adj_mask = self.bias_mat < -1.

        # --- Model properties ------------------------------------------------
        with TimeScope(MODEL_NAME + "/prop_setup", debug_only=True):
            self.inference
            self.loss
            # self.optimize

    def get_feed_dict(self, x_batch, y_batch, is_training):
        xb_node_feats = [np.array(x_i[0]) for x_i in x_batch]
        xb_bias_mat = [np.array(x_i[1]) for x_i in x_batch]
        xb_tconv_index = [np.array(x_i[2]) for x_i in x_batch]
        xb_valid_pts = [np.diag(x_i[3]) for x_i in x_batch]

        pool_drop = p.pool_drop_prob if is_training else 0.

        for batch_idx in range(len(xb_tconv_index)):
            xb_tconv_index[batch_idx] += batch_idx*p.nodes_nb

        return {
            self.node_feats: xb_node_feats,
            self.bias_mat: xb_bias_mat,
            self.tconv_index: np.array(xb_tconv_index).reshape((-1, 9)),
            self.valid_pts: xb_valid_pts,
            self.y: y_batch,
            self.pool_drop: pool_drop,
            self.is_training: is_training
        }

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """

        # --- Features dim reduction ------------------------------------------
        feat_red_out = self.node_feats
        with tf.variable_scope('feat_dim_red'):
            feat_red_out = tf.reshape(feat_red_out, [-1, p.feat_nb[0],
                                                     p.feat_nb[0],
                                                     self.num_channels])

            for i in range(0, len(p.red_hid_units), 2):
                feat_red_out = conv2d_bn(feat_red_out,
                                         out_sz=p.red_hid_units[i],
                                         kernel_sz=(3, 3),
                                         reg_constant=p.reg_constant,
                                         scope="conv_" + str(i),
                                         is_training=self.is_training)
                feat_red_out = conv2d_bn(feat_red_out,
                                         out_sz=p.red_hid_units[i+1],
                                         kernel_sz=(3, 3),
                                         reg_constant=p.reg_constant,
                                         scope="conv_" + str(i+1),
                                         is_training=self.is_training)
                feat_red_out = tf.layers.max_pooling2d(feat_red_out, 2, 2)

            feat_red_out = tf.reduce_max(feat_red_out, axis=1,
                                         name='max_u')
            feat_red_out = tf.reduce_max(feat_red_out, axis=1,
                                         name='max_v')

            feat_red_out = tf.reshape(feat_red_out,
                                      [-1, p.red_hid_units[-1]])

        # --- Graph attention layers ------------------------------------------
        with tf.variable_scope('tang_conv_layers'):
            # Pre setup
            feat_tconv = feat_red_out

            # Apply all convolutions
            for i in range(len(p.tconv_hid_units)):
                feat_tconv = point_conv("tang_conv_" + str(i),
                                        feat_tconv, self.tconv_index,
                                        filter_size=9,
                                        out_channels=p.tconv_hid_units[i])

            tconv_out = tf.reshape(feat_tconv,
                                   [-1, p.nodes_nb, p.tconv_hid_units[-1]])

        # --- Set Pooling -----------------------------------------------------
        with tf.variable_scope('graph_pool'):
            valid_pts = self.valid_pts

            tconv_filt = tf.matmul(valid_pts, tconv_out)
            max_gg = tf.reduce_max(tconv_filt, axis=1, name='max_g')
            fcg = fc_bn(max_gg, p.tconv_hid_units[-1],
                        scope='fcg',
                        is_training=self.is_training,
                        bn_decay=self.bn_decay,
                        reg_constant=p.reg_constant)

        # --- Classification --------------------------------------------------
        with tf.variable_scope('classification'):
            fc_2 = fc_bn(fcg, 128, scope='fc_2',
                         is_training=self.is_training,
                         bn_decay=self.bn_decay,
                         reg_constant=p.reg_constant)
            fc_2 = tf.nn.dropout(fc_2, 1.0 - self.pool_drop)

            return fc(fc_2, p.num_classes,
                      activation_fn=None, scope='logits')

    @define_scope
    def loss(self):
        #  --- Cross-entropy loss ---------------------------------------------
        with tf.variable_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=self.inference)

            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy_avg', cross_entropy)

        # --- L2 Regularization -----------------------------------------------
        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss_avg', reg_loss)

        total_loss = cross_entropy + reg_loss

        tf.summary.scalar('total_loss', total_loss)
        return total_loss
