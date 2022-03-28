#pragma once

#include "parameters.h"
#include "augmentation_preprocessing.cpp"
#include "occupancy.cpp"
#include "scope_time.h"

typedef pcl::PointXYZINormal PointT;
typedef std::pair<uint, uint> Edge;


class GraphConstructor
{
protected:
  std::string filename_;
  pcl::PointCloud<PointT>::Ptr pc_;
  pcl::PolygonMesh::Ptr mesh_;
  pcl::search::KdTree<PointT>::Ptr tree_;
  std::vector<std::vector<int> > nodes_elts_;
  std::vector<std::vector<int> > nodes_vertices_;
  std::vector<std::vector<int> > node_face_association_;
  std::vector<std::vector<bool> > node_vertex_association_;
  std::unordered_map<Edge, std::array<int, 2>, boost::hash<Edge> > edge_to_triangle_;
  std::vector<std::vector<uint> > triangle_neighbors_;
  std::vector<int> sampled_indices_;
  std::vector<bool> valid_indices_;
  std::vector<std::vector<std::vector<int> > > lut_;
  std::vector<Eigen::Matrix3f> lrf_;
  std::vector<Eigen::Matrix3f> zlrf_;
  std::vector<Eigen::Vector3f> nodes_mean_;
  std::vector<Eigen::Vector3f> face_normals_;
  std::vector<float> face_angle_;
  std::vector<std::vector<uint> > boundary_loops_;
  double scale_;

  // Parameters
  unsigned int gridsize_;
  unsigned int nodes_nb_;
  bool debug_;


  void computeNormalAlignedPcaLrf();
  void computeVerticesCurvature();
  void areaBasedNodeSampling(float target_area);
  void angleBasedNodeSampling(float target_angle);
  void extractNodeBoundaries();

public:
  GraphConstructor(std::string filename,
                   unsigned int gridsize,
                   unsigned int nodes_nb,
                   bool debug) :
    filename_(filename),
    pc_(new pcl::PointCloud<PointT>),
    mesh_(new pcl::PolygonMesh),
    tree_(new pcl::search::KdTree<PointT>),
    gridsize_(gridsize),
    nodes_nb_(nodes_nb),
    debug_(debug) {
      // Reserve space for the neighborhood of nodes
      nodes_elts_.reserve(nodes_nb_);

      // Initialize the valid indices
      for (uint i=0; i<nodes_nb_; i++)
        valid_indices_.push_back(false);

    }

  // General
  void initializeMesh();
  void initializeMesh(float* vertices, uint vertex_nb, int* triangles, uint triangle_nb);
  void initializeMesh(float* vertices, uint vertex_nb, int* triangles, uint triangle_nb, float* normals);
  int initializeParts(bool angle_sampling, double* adj_mat, float neigh_size);
  void correctAdjacencyForValidity(double* adj_mat);
  void getValidIndices(int* valid_indices);
  void vizGraph(double* adj_mat, VizParams viz_params);

  // Node features
  void lEsfNodeFeatures(double** result, unsigned int feat_nb);
  void coordsSetNodeFeatures(double** result, unsigned int feat_nb, bool use_zlrf, unsigned int num_channels);
  void sphNodeFeatures(double** result, int* tconv_idx, uint image_size, uint num_channels, SphParams sph_params);
  void pointProjNodeFeatures(double** result, int* tconv_idx, uint image_size);

  // Adjacency matrix construction method
  void fullConnectionAdjacency(double* adj_mat);
  void occupancyAdjacency(double* adj_mat, unsigned int neigh_nb);

  // Edge features
  void coordsEdgeFeatures(double* edge_feats);
  void rotZEdgeFeatures(double* edge_feats, float min_angle_z_normal);
  void tconvEdgeFeatures(int* tconv_idx, double* tconv_angle);
};
