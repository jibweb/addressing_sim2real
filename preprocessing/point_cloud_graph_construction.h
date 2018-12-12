#pragma once

#include <pcl/io/pcd_io.h>

#include "parameters.h"
#include "augmentation_preprocessing.cpp"
#include "occupancy.cpp"
#include "scope_time.h"

class PointCloudGraphConstructor
{
protected:
  std::string filename_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_;
  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree_;
  std::vector<std::vector<int> > nodes_elts_;
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
  PointCloudGraphConstructor(std::string filename,
                             Parameters params,
                             unsigned int gridsize,
                             unsigned int nodes_nb,
                             bool debug) :
    filename_(filename),
    pc_(new pcl::PointCloud<pcl::PointXYZINormal>),
    tree_(new pcl::search::KdTree<pcl::PointXYZINormal>),
    gridsize_(gridsize),
    nodes_nb_(nodes_nb),
    debug_(debug),
    params_(params) {}

  void initialize() {
    ScopeTime t("Initialization (PointCloudGraphConstructor)", debug_);

    // Read the point cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (filename_.c_str(), *pc_) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
      return;
    }

    // Data augmentation
    Eigen::Vector4f centroid;
    scale_ = scale_points_unit_sphere (*pc_, gridsize_/2, centroid);
    params_.neigh_size = params_.neigh_size * gridsize_/2;
    augment_data(pc_, params_, gridsize_, debug_);

    if (params_.scale && debug_)
      std::cout << "Scale: " << scale_ << std::endl;


    // Initialize the tree
    tree_->setInputCloud (pc_);


    // Initialize the valid indices
    for (uint i=0; i<nodes_nb_; i++)
      valid_indices_.push_back(false);


    // Prepare the voxel grid
    lut_.resize (gridsize_);
    for (uint i = 0; i < gridsize_; ++i) {
        lut_[i].resize (gridsize_);
        for (uint j = 0; j < gridsize_; ++j)
          lut_[i][j].resize (gridsize_);
    }

    voxelize (*pc_, lut_, gridsize_);
  };

  // General
  void samplePoints();
  void correctAdjacencyForValidity(double* adj_mat);
  void getValidIndices(int* valid_indices);
  void viz(double* adj_mat, bool viz_small_spheres);

  // node features
  void lEsfNodeFeatures(double** result, unsigned int feat_nb);
  void esf3dNodeFeatures(double** result);
  void coordsSetNodeFeatures(double** result, unsigned int feat_nb);

  // Adjacency matrix construction method
  void fullConnectionAdjacency(double* adj_mat);
  void occupancyAdjacency(double* adj_mat, unsigned int neigh_nb);

  // Edge features
  void coordsEdgeFeatures(double* edge_feats);
  void rotZEdgeFeatures(double* edge_feats);
};
