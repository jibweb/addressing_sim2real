import numpy as np
import tensorflow as tf
from utils.logger import TimeScope
from utils.tf import fc, fc_bn, define_scope
from utils.params import params as p

from layers import conv2d_bn, point_conv, conv1d_bn, g_2d_k, attn_head


MODEL_NAME = "COORDSSET_GAT_MaxPool"

# # Dropout prob params
# p.define("attn_drop_prob", 0.0)
# p.define("feat_drop_prob", 0.0)
# p.define("pool_drop_prob", 0.5)
# # Model arch params
# p.define("residual", False)
# p.define("graph_hid_units", [16])
# p.define("attn_head_nb", [16])
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

            self.edge_feats = tf.placeholder(tf.float32,
                                             (None,
                                              p.nodes_nb,
                                              p.nodes_nb,
                                              p.edge_feat_nb),
                                             name="edge_feats")

            self.valid_pts = tf.placeholder(tf.float32,
                                            [None,
                                             p.nodes_nb,
                                             p.nodes_nb],
                                            name="valid_pts")

            self.node_feats = tf.placeholder(tf.float32,
                                             (None,
                                              p.nodes_nb,
                                              p.feat_nb[0],
                                              p.feat_nb[1]),
                                             name="node_feats")

            self.y = tf.placeholder(tf.float32,
                                    [None, p.num_classes],
                                    name="y")
            self.mask = tf.placeholder(tf.float32,
                                       [None],
                                       name="mask")
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
        xb_edge_feats = [np.array(x_i[2]) for x_i in x_batch]
        xb_valid_pts = [np.diag(x_i[3]) for x_i in x_batch]
        xb_mask = [item for x_i in x_batch for item in x_i[3]]

        y_batch = [val for val in y_batch for i in range(p.nodes_nb)]

        pool_drop = p.pool_drop_prob if is_training else 0.

        return {
            self.node_feats: xb_node_feats,
            self.bias_mat: xb_bias_mat,
            self.edge_feats: xb_edge_feats,
            self.valid_pts: xb_valid_pts,
            self.y: y_batch,
            self.pool_drop: pool_drop,
            self.is_training: is_training,
            self.mask: xb_mask
        }

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """

        # --- Features dim reduction ------------------------------------------
        feat_red_out = self.node_feats

        print "a",  feat_red_out.get_shape()

        with tf.variable_scope('feat_dim_red'):
            for i in range(len(p.red_hid_units)):
                feat_red_out = g_2d_k(feat_red_out,
                                      "g2d_" + str(i),
                                      p.red_hid_units[i],
                                      self.is_training, self.bn_decay,
                                      p.reg_constant)

            print "b",  feat_red_out.get_shape()

            feat_red_out = tf.reduce_max(feat_red_out, axis=2,
                                         name='max_g')

            print "c",  feat_red_out.get_shape()
            feat_red_out = conv1d_bn(feat_red_out,
                                     scope="fc_1",
                                     out_sz=p.red_hid_units[-1],
                                     reg_constant=p.reg_constant,
                                     is_training=self.is_training)

            print "d1",  feat_red_out.get_shape()
            feat_red_out = conv1d_bn(feat_red_out,
                                     scope="fc_2",
                                     out_sz=p.red_hid_units[-1]/2,
                                     reg_constant=p.reg_constant,
                                     is_training=self.is_training)

            print "d2",  feat_red_out.get_shape()

            feat_red_out = tf.reshape(feat_red_out,
                                      [-1, p.nodes_nb, p.red_hid_units[-1]/2])

            print "e",  feat_red_out.get_shape()

        # --- Graph attention layers ------------------------------------------
        with tf.variable_scope('graph_layers'):
            # Pre setup
            feat_gcn = feat_red_out
            edge_feats = self.edge_feats
            bias_mat = self.bias_mat

            print "GCNa",  feat_gcn.get_shape()

            # Apply all convolutions
            for i in range(len(p.graph_hid_units)):
                gcn_heads = []
                for head_idx in range(p.attn_head_nb[i]):
                    head = attn_head(feat_gcn,
                                     out_sz=p.graph_hid_units[i],
                                     bias_mat=bias_mat,
                                     activation=tf.nn.elu,
                                     reg_constant=p.reg_constant,
                                     is_training=self.is_training,
                                     bn_decay=self.bn_decay,
                                     scope="attn_heads_" + str(i) + "/head_" + str(head_idx))
                    gcn_heads.append(head)

                feat_gcn = tf.concat(gcn_heads, axis=-1)

            gcn_out = feat_gcn

        # --- Classification --------------------------------------------------
        with tf.variable_scope('classification'):
            valid_pts = self.valid_pts
            gcn_filt = tf.matmul(valid_pts, gcn_out)
            gcn_reshaped = tf.reshape(gcn_filt,
                                      [-1, p.graph_hid_units[-1]*p.attn_head_nb[-1]])

            fcg = fc_bn(gcn_reshaped, p.graph_hid_units[-1]*p.attn_head_nb[-1],
                        scope='fcg',
                        is_training=self.is_training,
                        bn_decay=self.bn_decay,
                        reg_constant=p.reg_constant)

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
            diff = tf.boolean_mask(diff, self.mask, name='boolean_mask')

            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy_avg', cross_entropy)

        # --- L2 Regularization -----------------------------------------------
        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss_avg', reg_loss)

        total_loss = cross_entropy + reg_loss

        tf.summary.scalar('total_loss', total_loss)
        return total_loss
