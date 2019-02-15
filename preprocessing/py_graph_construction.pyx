import cython
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


cdef extern from "graph_construction.cpp":
    pass

# Decalre the class with cdef
cdef extern from "graph_construction.h":
    ctypedef struct Parameters:
        # Graph structure
        # unsigned int nodes_nb
        # unsigned int feat_nb
        # unsigned int edge_feat_nb
        # float min_angle_z_normal
        # float neigh_size
        # int neigh_nb
        # bint feats_3d
        # bint edge_feats
        # bint mesh
        bint scale
        # General
        # unsigned int gridsize
        # bint viz
        # bint viz_small_spheres
        # bint debug
        # PC transformations
        float to_remove
        unsigned int to_keep
        float occl_pct
        float noise_std
        unsigned int rotation_deg
    cdef cppclass GraphConstructor:
        GraphConstructor(string filename, Parameters,
                         unsigned int gridsize,
                         unsigned int nodes_nb,
                         bool debug) except +

        # General
        # void initialize()
        void initializePointCloud(float min_angle_z_normal, float neigh_size)
        void initializeMesh(float min_angle_z_normal, double* adj_mat, float neigh_size)
        void correctAdjacencyForValidity(double* adj_mat)
        void getValidIndices(int* valid_indices)
        void viz(double* adj_mat, bool)
        void vizMesh(double* adj_mat, bool)

        # Node features
        void lEsfNodeFeatures(double** result, unsigned int)
        void sphNodeFeatures(double** result, unsigned int image_size, unsigned int r_sdiv, unsigned int p_sdiv)

        # Adjacency construction
        void fullConnectionAdjacency(double* adj_mat)
        void occupancyAdjacency(double* adj_mat, unsigned int)

        # Edge features
        void coordsEdgeFeatures(double* edge_feats)
        void rotZEdgeFeatures(double* edge_feats, float min_angle_z_normal)


cdef class PyGraph:
    cdef GraphConstructor*c_graph  # Hold a C++ instance which we're wrapping
    cdef unsigned int nodes_nb

    def __cinit__(self, string fn, nodes_nb=16, debug=True, gridsize=64):
        cdef Parameters params
        # params.feat_nb = 800
        # params.edge_feat_nb = 3
        # params.min_angle_z_normal = 10
        # params.neigh_size = 0.401
        # params.neigh_nb = 4
        # params.feats_3d = True
        # params.edge_feats = True
        # params.mesh = False
        # params.viz = False
        # params.viz_small_spheres = True
        params.to_remove = 0.
        params.to_keep = 20000
        params.occl_pct = 0.
        params.noise_std = 0.
        params.rotation_deg = 0

        self.nodes_nb = nodes_nb
        self.c_graph = new GraphConstructor(fn, params, gridsize, nodes_nb, debug)

    def initialize_point_cloud(self, float min_angle_z_normal, float neigh_size):
        self.c_graph.initializePointCloud(min_angle_z_normal, neigh_size)

    def initialize_mesh(self, float min_angle_z_normal, float neigh_size):
        cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([self.nodes_nb,
                                                                      self.nodes_nb],
                                                                     dtype=np.float64)
        self.c_graph.initializeMesh(min_angle_z_normal, &adj_mat[0, 0], neigh_size)
        return adj_mat

    # def sample_points(self, float min_angle_z_normal):
    #     self.c_graph.samplePoints(min_angle_z_normal)

    def correct_adjacency_for_validity(self, np.ndarray[np.float64_t, ndim=2] adj_mat):
        self.c_graph.correctAdjacencyForValidity(&adj_mat[0, 0])

    def get_valid_indices(self):
        cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = np.zeros([self.nodes_nb],
                                                                        dtype=np.int32)
        self.c_graph.getValidIndices(&valid_indices[0])
        return valid_indices

    def viz(self, np.ndarray[np.float64_t, ndim=2] adj_mat, bool viz_small_spheres=True):
        self.c_graph.viz(&adj_mat[0, 0], viz_small_spheres)

    def viz_mesh(self, np.ndarray[np.float64_t, ndim=2] adj_mat, bool viz_small_spheres=True):
        self.c_graph.vizMesh(&adj_mat[0, 0], viz_small_spheres)

    def node_features_l_esf(self, feat_nb):
        """
        """
        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        node_feats2d = []
        cdef np.ndarray[double, ndim=2, mode="c"] tmp

        for i in range(self.nodes_nb):
            tmp = np.zeros([feat_nb, 6], dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0]
            node_feats2d.append(tmp)

        self.c_graph.lEsfNodeFeatures(node_feats2d_ptr, feat_nb)
        return node_feats2d

    def node_features_sph(self, image_size, r_sdiv, p_sdiv):
        """
        """
        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        node_feats2d = []
        cdef np.ndarray[double, ndim=3, mode="c"] tmp

        for i in range(self.nodes_nb):
            tmp = np.zeros([r_sdiv, p_sdiv, 3], dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0, 0]
            node_feats2d.append(tmp)

        self.c_graph.sphNodeFeatures(node_feats2d_ptr, image_size, r_sdiv, p_sdiv)
        return node_feats2d

    def adjacency_full_connection(self):
        cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([self.nodes_nb,
                                                                      self.nodes_nb],
                                                                     dtype=np.float64)
        self.c_graph.fullConnectionAdjacency(&adj_mat[0, 0])
        return adj_mat

    def adjacency_occupancy(self, neigh_nb):
        cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([self.nodes_nb,
                                                                      self.nodes_nb],
                                                                     dtype=np.float64)
        self.c_graph.occupancyAdjacency(&adj_mat[0, 0], neigh_nb)
        return adj_mat

    def edge_features_coords(self):
        cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([self.nodes_nb,
                                                                             self.nodes_nb,
                                                                             3],
                                                                            dtype=np.float64)
        self.c_graph.coordsEdgeFeatures(&edge_feats_mat[0, 0, 0])
        return edge_feats_mat

    def edge_features_rot_z(self, float min_angle_z_normal):
        """
        """
        cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([self.nodes_nb,
                                                                             self.nodes_nb,
                                                                             3],
                                                                            dtype=np.float64)
        self.c_graph.rotZEdgeFeatures(&edge_feats_mat[0, 0, 0], min_angle_z_normal)
        return edge_feats_mat

    def __dealloc__(self):
        del self.c_graph


# # declare the interface to the C code
# cdef extern from "wrapper_interface.cpp":
#     ctypedef struct Parameters:
#         # Graph structure
#         unsigned int nodes_nb
#         unsigned int feat_nb
#         unsigned int edge_feat_nb
#         float min_angle_z_normal
#         float neigh_size
#         int neigh_nb
#         bint feats_3d
#         bint edge_feats
#         bint mesh
#         bint scale
#         # General
#         unsigned int gridsize
#         bint viz
#         bint viz_small_spheres
#         bint debug
#         # PC transformations
#         float to_remove
#         unsigned int to_keep
#         float occl_pct
#         float noise_std
#         unsigned int rotation_deg

#     int construct_graph(string filename,
#                         double* node_feats,
#                         double* adj_mat,
#                         double* edge_feats,
#                         int* valid_indices,
#                         Parameters params)

#     int construct_graph_nd(string filename,
#                            double** node_feats,
#                            double* adj_mat,
#                            double* edge_feats,
#                            int* valid_indices,
#                            Parameters params)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def get_graph(filename, **kwargs):
#     cdef Parameters params
#     params.nodes_nb = kwargs.get("nodes_nb")
#     params.feat_nb = kwargs.get("feat_nb")
#     params.edge_feat_nb = kwargs.get("edge_feat_nb")
#     params.neigh_size = kwargs.get("neigh_size")
#     params.neigh_nb = kwargs.get("neigh_nb")
#     params.feats_3d = kwargs.get("feats_3d")
#     params.edge_feats = kwargs.get("edge_feats")
#     params.mesh = kwargs.get("mesh")
#     params.scale = kwargs.get("scale")

#     params.gridsize = kwargs.get("gridsize")
#     params.viz = kwargs.get("viz")
#     params.viz_small_spheres = kwargs.get("viz_small_spheres")
#     params.debug = kwargs.get("debug")

#     params.to_remove = kwargs.get("to_remove")
#     params.to_keep = kwargs.get("to_keep")
#     params.occl_pct = kwargs.get("occl_pct")
#     params.noise_std = kwargs.get("noise_std")
#     params.rotation_deg = kwargs.get("rotation_deg")
#     params.min_angle_z_normal = kwargs.get("min_angle_z_normal")

#     cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
#                                                                   params.nodes_nb],
#                                                                  dtype=np.float64)

#     cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([params.nodes_nb,
#                                                                          params.nodes_nb,
#                                                                          params.edge_feat_nb],
#                                                                         dtype=np.float64)

#     cdef np.ndarray[double, ndim=2, mode="c"] node_feats = np.zeros([params.nodes_nb,
#                                                                      params.feat_nb],
#                                                                     dtype=np.float64)

#     cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = np.zeros([params.nodes_nb],
#                                                                     dtype=np.int32)

#     if params.debug:
#         print "### File:", filename

#     construct_graph(filename, &node_feats[0, 0], &adj_mat[0, 0], &edge_feats_mat[0, 0, 0], &valid_indices[0], params)
#     return node_feats, adj_mat, edge_feats_mat, valid_indices


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def get_graph_nd(filename, **kwargs):
#     cdef Parameters params
#     params.nodes_nb = kwargs.get("nodes_nb")
#     params.feat_nb = kwargs.get("feat_nb")
#     params.edge_feat_nb = kwargs.get("edge_feat_nb")
#     params.neigh_size = kwargs.get("neigh_size")
#     params.neigh_nb = kwargs.get("neigh_nb")
#     params.feats_3d = kwargs.get("feats_3d")
#     params.edge_feats = kwargs.get("edge_feats")
#     params.mesh = kwargs.get("mesh")
#     params.scale = kwargs.get("scale")

#     params.gridsize = kwargs.get("gridsize")
#     params.viz = kwargs.get("viz")
#     params.viz_small_spheres = kwargs.get("viz_small_spheres")
#     params.debug = kwargs.get("debug")

#     params.to_remove = kwargs.get("to_remove")
#     params.to_keep = kwargs.get("to_keep")
#     params.occl_pct = kwargs.get("occl_pct")
#     params.noise_std = kwargs.get("noise_std")
#     params.rotation_deg = kwargs.get("rotation_deg")
#     params.min_angle_z_normal = kwargs.get("min_angle_z_normal")

#     cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
#                                                                   params.nodes_nb],
#                                                                  dtype=np.float64)

#     cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([params.nodes_nb,
#                                                                          params.nodes_nb,
#                                                                          params.edge_feat_nb],
#                                                                         dtype=np.float64)

#     if params.debug:
#         print "\n###\n File:", filename
#         print params

#     if params.feat_nb >= 500:
#         node_shape = [params.feat_nb, 6, 1]  # TODO TMP CHANGE !!
#     else:
#         node_shape = [params.feat_nb, params.feat_nb, params.feat_nb]

#     cdef double **node_feats3d_ptr = <double **> malloc(params.nodes_nb*sizeof(double *))
#     node_feats3d = []
#     cdef np.ndarray[double, ndim=3, mode="c"] temp

#     for i in range(params.nodes_nb):
#         temp = np.zeros(node_shape,
#                         dtype=np.float64)
#         node_feats3d_ptr[i] = &temp[0, 0, 0]

#         node_feats3d.append(temp)

#     cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = np.zeros([params.nodes_nb],
#                                                                     dtype=np.int32)

#     construct_graph_nd(filename, node_feats3d_ptr, &adj_mat[0, 0], &edge_feats_mat[0, 0, 0], &valid_indices[0], params)
#     return node_feats3d, adj_mat, edge_feats_mat, valid_indices