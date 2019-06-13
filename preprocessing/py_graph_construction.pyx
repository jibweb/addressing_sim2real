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
    ctypedef struct SphParams:
        bint mask
        bint plane_distance
        bint euclidean_distance
        bint z_height
        bint z_rel
        bint x_coords
        bint y_coords
        bint tconv_idx
        bint lscm
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
        void initializeMesh(float min_angle_z_normal, double* adj_mat, float neigh_size)
        void correctAdjacencyForValidity(double* adj_mat)
        void getValidIndices(int* valid_indices)
        void vizMesh(double* adj_mat, bool)

        # Node features
        void lEsfNodeFeatures(double** result, unsigned int)
        void coordsSetNodeFeatures(double** result, int* tconv_idx, unsigned int feat_nb, unsigned int num_channels)
        void sphNodeFeatures(double** result, int* tconv_idx, unsigned int image_size,  unsigned int num_channels, SphParams)
        void pointProjNodeFeatures(double** result, int* tconv_idx, unsigned int image_size)

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

    def node_features_sph(self, image_size, sph_config):
        """
        """
        num_channels = 0
        for key, val in sph_config.iteritems():
            if key == "lscm":
                continue
            if val:
                num_channels += 1

        cdef SphParams config_struct
        config_struct.mask = sph_config["mask"]
        config_struct.plane_distance = sph_config["plane_distance"]
        config_struct.euclidean_distance = sph_config["euclidean_distance"]
        config_struct.z_height = sph_config["z_height"]
        config_struct.z_rel = sph_config["z_rel"]
        config_struct.x_coords = sph_config["x_coords"]
        config_struct.y_coords = sph_config["y_coords"]
        config_struct.lscm = sph_config["lscm"]
        config_struct.tconv_idx = False

        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        cdef np.ndarray[int, ndim=2, mode="c"] tconv_indices = np.zeros([1, 1],
                                                                        dtype=np.int32)
        node_feats2d = []
        cdef np.ndarray[double, ndim=3, mode="c"] tmp
        arr_shape = [image_size, image_size, num_channels]

        for i in range(self.nodes_nb):
            tmp = np.zeros(arr_shape, dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0, 0]
            node_feats2d.append(tmp)

        self.c_graph.sphNodeFeatures(node_feats2d_ptr, &tconv_indices[0, 0],
                                     image_size, num_channels, config_struct)
        return node_feats2d

    def node_features_sph_tconv_idx(self, image_size, sph_config):
        """
        """
        num_channels = 0
        for key, val in sph_config.iteritems():
            if key == "lscm":
                continue
            if val:
                num_channels += 1

        cdef SphParams config_struct
        config_struct.mask = sph_config["mask"]
        config_struct.plane_distance = sph_config["plane_distance"]
        config_struct.euclidean_distance = sph_config["euclidean_distance"]
        config_struct.z_height = sph_config["z_height"]
        config_struct.z_rel = sph_config["z_rel"]
        config_struct.x_coords = sph_config["x_coords"]
        config_struct.lscm = sph_config["lscm"]
        config_struct.tconv_idx = True

        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        cdef np.ndarray[int, ndim=2, mode="c"] tconv_indices = np.zeros([self.nodes_nb, 9],
                                                                        dtype=np.int32)
        node_feats2d = []
        cdef np.ndarray[double, ndim=3, mode="c"] tmp
        arr_shape = [image_size, image_size, num_channels]

        for i in range(self.nodes_nb):
            tmp = np.zeros(arr_shape, dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0, 0]
            node_feats2d.append(tmp)

        self.c_graph.sphNodeFeatures(node_feats2d_ptr, &tconv_indices[0, 0],
                                     image_size, num_channels, config_struct)
        return node_feats2d, tconv_indices

    def node_features_pt_proj_tconv_idx(self, image_size):
        """
        """
        num_channels = 2
        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        cdef np.ndarray[int, ndim=2, mode="c"] tconv_indices = np.zeros([self.nodes_nb, 9],
                                                                        dtype=np.int32)
        node_feats2d = []
        cdef np.ndarray[double, ndim=3, mode="c"] tmp
        arr_shape = [image_size, image_size, num_channels]

        for i in range(self.nodes_nb):
            tmp = np.zeros(arr_shape, dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0, 0]
            node_feats2d.append(tmp)

        self.c_graph.pointProjNodeFeatures(node_feats2d_ptr,
                                           &tconv_indices[0, 0],
                                           image_size)
        return node_feats2d, tconv_indices

    def node_features_coords_set_tconv_idx(self, feat_nb, num_channels):
        """
        """
        cdef double **node_feats2d_ptr = <double **> malloc(self.nodes_nb*sizeof(double *))
        cdef np.ndarray[int, ndim=2, mode="c"] tconv_indices = np.zeros([self.nodes_nb, 9],
                                                                        dtype=np.int32)
        node_feats2d = []
        cdef np.ndarray[double, ndim=2, mode="c"] tmp
        arr_shape = [feat_nb, num_channels]

        for i in range(self.nodes_nb):
            tmp = np.zeros(arr_shape, dtype=np.float64)
            node_feats2d_ptr[i] = &tmp[0, 0]
            node_feats2d.append(tmp)

        self.c_graph.coordsSetNodeFeatures(node_feats2d_ptr,
                                           &tconv_indices[0, 0],
                                           feat_nb,
                                           num_channels)
        return node_feats2d, tconv_indices

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
