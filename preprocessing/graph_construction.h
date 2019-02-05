#pragma once

#include <igl/boundary_loop.h>
#include <igl/lscm.h>

#include "parameters.h"
#include "augmentation_preprocessing.cpp"
#include "occupancy.cpp"
#include "scope_time.h"

typedef pcl::PointXYZINormal PointT;


class GraphConstructor
{
protected:
  std::string filename_;
  pcl::PointCloud<PointT>::Ptr pc_;
  pcl::PolygonMesh::Ptr mesh_;
  pcl::search::KdTree<PointT>::Ptr tree_;
  std::vector<std::vector<int> > nodes_elts_;
  std::vector<std::unordered_map<uint, std::vector<uint> > > node_surface_tree_;
  std::vector<std::unordered_map<uint, uint> > node_surface_tree_depth_;
  std::vector<std::vector<int> > adj_list_;
  std::vector<int> sampled_indices_;
  std::vector<bool> valid_indices_;
  std::vector<std::vector<std::vector<int> > > lut_;
  double scale_;

  // Parameters
  unsigned int gridsize_;
  unsigned int nodes_nb_;
  bool debug_;

  Parameters params_;

public:
  GraphConstructor(std::string filename,
                   Parameters params,
                   unsigned int gridsize,
                   unsigned int nodes_nb,
                   bool debug) :
    filename_(filename),
    pc_(new pcl::PointCloud<PointT>),
    mesh_(new pcl::PolygonMesh),
    tree_(new pcl::search::KdTree<PointT>),
    gridsize_(gridsize),
    nodes_nb_(nodes_nb),
    debug_(debug),
    params_(params) {
      // Reserve space for the neighborhood of nodes
      nodes_elts_.reserve(nodes_nb_);

      // Initialize the valid indices
      for (uint i=0; i<nodes_nb_; i++)
        valid_indices_.push_back(false);

    }

  // General
  void initializePointCloud(float min_angle_z_normal, float neigh_size);
  void initializeMesh(float min_angle_z_normal, double* adj_mat, unsigned int neigh_nb);
  void correctAdjacencyForValidity(double* adj_mat);
  void getValidIndices(int* valid_indices);
  void viz(double* adj_mat, bool viz_small_spheres);
  void vizMesh(double* adj_mat, bool viz_small_spheres);

  // Node features
  void lEsfNodeFeatures(double** result, unsigned int feat_nb);
  // void esf3dNodeFeatures(double** result);
  void coordsSetNodeFeatures(double** result, unsigned int feat_nb);
  void sphNodeFeatures(double** result, uint image_size, uint r_sdiv, uint p_sdiv);

  // Adjacency matrix construction method
  void fullConnectionAdjacency(double* adj_mat);
  void occupancyAdjacency(double* adj_mat, unsigned int neigh_nb);

  // Edge features
  void coordsEdgeFeatures(double* edge_feats);
  void rotZEdgeFeatures(double* edge_feats, float min_angle_z_normal);
};
