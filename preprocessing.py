from enum import Enum
from functools import partial
import numpy as np
import trimesh

from utils.params import params as p
from py_graph_construction import PyGraph, NanNormals


VERTEX_FEAT = Enum("VERTEX_FEAT", "L_ESF SPH COORDSSET")
EDGE_FEAT = Enum("EDGE_FEAT", "ROT_Z COORDS TCONV")


# Graph structure
p.define("nodes_nb", 128)

# Vertices
p.define("feat_type", VERTEX_FEAT.SPH.name)
p.define("feat_nb", [100])
p.define("feat_config", {
        "mask": True,
        "plane_distance": True,
        "euclidean_distance": False,
        "z_height": True,
        "z_rel": False,
        "x_coords": False,
        "y_coords": False,
    })

# Edges
p.define("edge_feat_type", EDGE_FEAT.COORDS.name)
p.define("edge_feat_nb", 3)

p.define("neigh_size", 0.15)
p.define("gridsize", 64)
# p.define("scale", False)
p.define("min_angle_z_normal", 0)
p.define("angle_sampling", True)

# Data transformation
p.define("to_remove", 0.)
p.define("to_keep", 5000)
p.define("occl_pct", 0.)
p.define("noise_std", 0.)
p.define("rotation_deg", 0)

p.define("debug", False)
p.define("viz", False)
p.define("viz_small_spheres", False)


def preprocess_dummy(data):
    return data


def preprocess_adj_to_bias(adj):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    # mt = adj + np.eye(adj.shape[1])
    return -1e9 * (1.0 - adj)


# def graph_preprocess_shot(fn, p):
#     feats, adj = get_graph_feats(fn, **p.__dict__)
#     bias = adj_to_bias(adj)

#     # 2-hop adj matrix
#     # adj_2hop = np.matmul(adj, adj)
#     # adj_2hop = (adj_2hop > 0).astype(adj_2hop.dtype)
#     # bias_2hop = adj_to_bias(adj_2hop)

#     return feats, bias


def preprocess_fpfh(feats):
    max_feats = np.max(feats, axis=1) + 1e-6
    feats = feats / np.repeat(max_feats.reshape((p.nodes_nb, 1)), 33, axis=1)
    return feats


def preprocess_esf3d(feats):
    return np.array(feats)[..., np.newaxis]


def preprocess_lesf(feats):
    return np.array(feats)[..., 0]


# def graph_preprocess_3d(fn, p, preprocess_feats, preprocess_adj,
#                         preprocess_edge_feats):
#     feats, adj, edge_feats, valid_indices = get_graph_nd(fn, **p.__dict__)
#     feats = preprocess_feats(feats)
#     adj = preprocess_adj(adj)
#     edge_feats = preprocess_edge_feats(edge_feats)

#     return feats, adj, edge_feats, valid_indices


# def graph_preprocess(fn, p, preprocess_feats, preprocess_adj,
#                      preprocess_edge_feats):
#     try:
#         feats, adj, edge_feats, valid_indices = get_graph(fn, **p.__dict__)
#     except:
#         print fn
#         return

#     feats = preprocess_feats(feats)
#     adj = preprocess_adj(adj)
#     edge_feats = preprocess_edge_feats(edge_feats)

#     return feats, adj, edge_feats, valid_indices

def graph_preprocess_new(fn, p, edge_feat, vertex_feat, with_fn):
    gc = PyGraph(fn, nodes_nb=p.nodes_nb, debug=p.debug, gridsize=p.gridsize)
    # gc.initialize()
    # gc.sample_points(p.min_angle_z_normal)
    gc.initialize_mesh_from_file()

    # ------------------------ OCCL EXP ---------------------------------------
    # mesh = trimesh.load_mesh(fn)
    # occlusion = 0.6

    # if occlusion:
    #     min_occl = float(occlusion)
    #     adjacency = [[] for _ in range(len(mesh.faces))]
    #     for (e1, e2) in mesh.face_adjacency:
    #         adjacency[e1].append(e2)
    #         adjacency[e2].append(e1)

    #     start_face = np.random.randint(len(mesh.faces))
    #     to_rm = set([start_face])
    #     queue = [start_face]
    #     occl = float(len(to_rm)) / len(mesh.faces)
    #     while occl < min_occl and len(queue) > 0:
    #         cur_idx = queue.pop(0)
    #         for f in adjacency[cur_idx]:
    #             if f not in to_rm:
    #                 queue.append(f)
    #                 to_rm.add(f)

    #         occl = float(len(to_rm)) / len(mesh.faces)

    #     to_keep = [idx for idx in range(len(mesh.faces)) if not idx in to_rm]
    #     mesh = trimesh.util.submesh(mesh, [to_keep])[0]

    # gc.initialize_mesh_from_array(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32))
    # ------------------------ /OCCL EXP --------------------------------------

    adj_mat = gc.initialize_parts(p.angle_sampling, p.neigh_size)

    if p.edge_feat_type == edge_feat.ROT_Z.name:
        edge_feats = gc.edge_features_rot_z(p.min_angle_z_normal)
    elif p.edge_feat_type == edge_feat.COORDS.name:
        edge_feats = gc.edge_features_coords()
    elif p.edge_feat_type == edge_feat.TCONV.name:
        edge_feats = gc.edge_features_tconv()

    if p.feat_type == vertex_feat.L_ESF.name:
        node_feats = gc.node_features_l_esf(p.feat_nb)
    elif p.feat_type == vertex_feat.SPH.name:
        node_feats = gc.node_features_sph(image_size=p.feat_nb[0],
                                          sph_config=p.feat_config)
    elif p.feat_type == vertex_feat.COORDSSET.name:
        node_feats = gc.node_features_coords_set(
                p.feat_nb[0], p.feat_config, p.feat_nb[1])

    gc.correct_adjacency_for_validity(adj_mat)
    valid_indices = gc.get_valid_indices()

    adj_mat = preprocess_adj_to_bias(adj_mat)

    if with_fn:
        return node_feats, adj_mat, edge_feats, valid_indices, fn
    else:
        return node_feats, adj_mat, edge_feats, valid_indices


def get_graph_preprocessing_fn(p, with_fn=False):

    edge_feat_names = [f.name for f in EDGE_FEAT]
    vertex_feat_names = [f.name for f in VERTEX_FEAT]

    try:
        assert p.edge_feat_type in edge_feat_names
    except AssertionError:
        raise Exception("This edge feature type does not exist !"
                        " Check the spelling of 'edge_feat_type' parameter")

    try:
        assert p.feat_type in vertex_feat_names
    except AssertionError:
        raise Exception("This (vertex) feature type does not exist !"
                        " Check the spelling of 'feat_type' parameter")

    return partial(graph_preprocess_new, p=p,
                   edge_feat=EDGE_FEAT, vertex_feat=VERTEX_FEAT,
                   with_fn=with_fn)
    # if p.feats_3d:
    #     if p.feat_nb == 4:
    #         return partial(graph_preprocess_3d, p=p,
    #                        preprocess_feats=preprocess_esf3d,
    #                        preprocess_adj=preprocess_adj_to_bias,
    #                        preprocess_edge_feats=preprocess_dummy)
    #     else:
    #         return partial(graph_preprocess_3d, p=p,
    #                        preprocess_feats=preprocess_lesf,
    #                        preprocess_adj=preprocess_adj_to_bias,
    #                        preprocess_edge_feats=preprocess_dummy)
    # else:
    #     if p.feat_nb == 33:
    #         return partial(graph_preprocess, p=p,
    #                        preprocess_feats=preprocess_fpfh,
    #                        preprocess_adj=preprocess_adj_to_bias,
    #                        preprocess_edge_feats=preprocess_dummy)
    #     else:
    #         return partial(graph_preprocess, p=p,
    #                        preprocess_feats=preprocess_dummy,
    #                        preprocess_adj=preprocess_adj_to_bias,
    #                        preprocess_edge_feats=preprocess_dummy)
