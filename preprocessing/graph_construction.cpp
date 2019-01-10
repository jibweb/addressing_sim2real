#include <random>
#include <math.h>
// #include <time>

#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


#include "graph_construction.h"



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// GENERAL ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializePointCloud(float min_angle_z_normal, float neigh_size) {
  // ScopeTime t("Point sampling computation", debug_);
  ScopeTime t("Initialization (PointCloudGraphConstructor)", debug_);

  // Read the point cloud
  if (pcl::io::loadPCDFile<PointT> (filename_.c_str(), *pc_) == -1) {
    PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
    return;
  }

  // Data augmentation
  scale_ = scale_points_unit_sphere (*pc_, gridsize_/2);
  neigh_size = neigh_size * gridsize_/2;
  augment_data(pc_, params_, gridsize_, debug_);

  if (params_.scale && debug_)
    std::cout << "Scale: " << scale_ << std::endl;

  // Initialize the tree
  tree_->setInputCloud (pc_);

  // Prepare the voxel grid
  lut_.resize (gridsize_);
  for (uint i = 0; i < gridsize_; ++i) {
      lut_[i].resize (gridsize_);
      for (uint j = 0; j < gridsize_; ++j)
        lut_[i][j].resize (gridsize_);
  }

  voxelize (*pc_, lut_, gridsize_);

  // --- Setup for the sampling -----------------------------------------------
  sampled_indices_.clear();
  sampled_indices_.reserve(nodes_nb_);
  // Prepare the values for the sampling procedure
  srand (static_cast<unsigned int> (time (0)));
  int rdn_weight, index;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  int total_weight = pc_->points.size();
  std::vector<bool> probs(pc_->points.size(), true);

  // --- Remove nodes too close to the z vector -------------------------------
  if (min_angle_z_normal > 0.) {
    Eigen::Vector3f z(0., 0., 1.);
    z.normalize();
    for (uint i=0; i<pc_->points.size(); i++) {
      Eigen::Vector3f n1 = pc_->points[i].getNormalVector3fMap();
      n1.normalize();

      if (acos(fabs(n1.dot(z))) < min_angle_z_normal*M_PI/180.) {
        probs[i] = false;
        total_weight -= 1;
      }
    }
  }

  // --- Sampling Procedure ---------------------------------------------------
  for (uint pt_idx=0; pt_idx < nodes_nb_; pt_idx++) {
    // Sample a new point
    if (total_weight > 0) {
      rdn_weight = rand() % total_weight;
      index = -1;
    } else {
      break;
    }

    for (uint i=0; i<pc_->points.size(); i++){
      if (!probs[i])
        continue;

      if (rdn_weight == 0) {
        index = i;
        break;
      }

      rdn_weight -= 1;
    }

    if (index == -1) {
      // There is no point left to sample !
      if (debug_)
        std::cout << "Couldn't sample " << nodes_nb_ - pt_idx << " salient points" << std::endl;
      break;
    }

    // Check if the sampled point is usable
    if (std::isnan(pc_->points[index].normal_x)) {
      probs[index] = false;
      total_weight -= 1;
      pt_idx--;
      continue;
    }

    if (neigh_size > 0.) {
      // Extract the sampled point neighborhood
      tree_->radiusSearch(pc_->points[index], neigh_size, k_indices, k_sqr_distances);
      nodes_elts_.push_back(k_indices);

      // Update the sampling probability
      for (uint i=0; i < k_indices.size(); i++) {
        if (probs[k_indices[i]])
          total_weight -= 1;

        probs[k_indices[i]] = false;
      }
    } else {
      probs[index] = false;
      total_weight -= 1;
    }

    sampled_indices_.push_back(index);
  }

  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++)
    valid_indices_[i] = true;

  if (debug_)
    std::cout << "Sampled points: " << sampled_indices_.size() << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializeMesh(float min_angle_z_normal, double* adj_mat, unsigned int neigh_nb) {

  ScopeTime t("Initialization (MeshGraphConstructor)", debug_);

  // Read the point cloud
  if (pcl::io::loadPLYFile(filename_.c_str(), *mesh_) == -1) {
    PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
    return;
  }


  std::string pc_filename = filename_.substr(0, filename_.size() - 4) + ".pcd";
  // Read the point cloud
  if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (pc_filename.c_str(), *pc_) == -1) {
    PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
    return;
  }


  if (debug_) {
    std::cout << "PolygonMesh: " << mesh_->polygons.size() << " triangles" << std::endl;
    std::cout << "PC size: " << pc_->points.size() << std::endl;
  }

  // Data augmentation
  // Eigen::Vector4f centroid;
  // scale_points_unit_sphere (*pc_, gridsize_/2, centroid);
  // params_.neigh_size = params_.neigh_size * gridsize_/2;
  // augment_data(pc_, params_);

  // Initialize the tree
  tree_->setInputCloud (pc_);

  nodes_elts_.resize(nodes_nb_);
  for (uint i=0; i < nodes_elts_.size(); i++)
    nodes_elts_[i].reserve(neigh_nb);


  // Allocate space for the adjacency list
  adj_list_.resize(pc_->points.size());
  for (uint i=0; i < adj_list_.size(); i++)
    adj_list_[i].reserve(16);


  // TODO Remove nodes with a wrong angle_z_normal

  // Fill in the adjacency list
  for (uint t=0; t < mesh_->polygons.size(); t++) {
    pcl::Vertices& triangle = mesh_->polygons[t];

    adj_list_[triangle.vertices[0]].push_back(triangle.vertices[1]);
    adj_list_[triangle.vertices[1]].push_back(triangle.vertices[0]);

    adj_list_[triangle.vertices[1]].push_back(triangle.vertices[2]);
    adj_list_[triangle.vertices[2]].push_back(triangle.vertices[1]);

    adj_list_[triangle.vertices[2]].push_back(triangle.vertices[0]);
    adj_list_[triangle.vertices[0]].push_back(triangle.vertices[2]);
  }



  // Initialize the valid indices
  valid_indices_.resize(nodes_nb_);
  // for (uint i=0; i<nodes_nb_; i++)
  //   valid_indices_.push_back(false);


  // Prepare the voxel grid
  lut_.resize (gridsize_);
  for (uint i = 0; i < gridsize_; ++i) {
      lut_[i].resize (gridsize_);
      for (uint j = 0; j < gridsize_; ++j)
        lut_[i][j].resize (gridsize_);
  }

  voxelize (*pc_, lut_, gridsize_);

  // --- BFS SAMPLING AND ADJACENCY -------------------------------------------
  // --- Global setup for the sampling procedure ------------------------------
  srand (static_cast<unsigned int> (time (0)));
  std::vector<bool> visited(pc_->points.size(), false);
  uint nb_visited = 0;
  std::vector<std::vector<int> > neighborhood(pc_->points.size());

  for (uint node_idx=0; node_idx < nodes_elts_.size(); node_idx++) {
    // --- Setup for BFS ------------------------------------------------------
    std::vector<bool> visited_local(pc_->points.size());
    uint nb_visited_local = 0;
    std::deque<int> queue;

    if (nb_visited == pc_->points.size())
      break;

    // --- Select a node and enqueue it ---------------------------------------
    int rdn_idx;
    do {
      rdn_idx = rand() % pc_->points.size();
    } while (visited[rdn_idx]);

    if (!visited[rdn_idx])
      nb_visited++;

    visited[rdn_idx] = true;
    visited_local[rdn_idx] = true;
    neighborhood[rdn_idx].push_back(node_idx);
    queue.push_back(rdn_idx);
    sampled_indices_.push_back(rdn_idx);


    // --- BFS over the graph to extract the neighborhood ---------------------
    while(!queue.empty() && nb_visited_local < neigh_nb)
    {
      // Dequeue a vertex from queue and print it
      int s = queue.front();
      queue.pop_front();
      nb_visited_local++;
      if (nb_visited_local < neigh_nb/2)
        nodes_elts_[node_idx].push_back(s);

      int nb_pt_idx;
      for (uint nb_idx=0; nb_idx < adj_list_[s].size(); nb_idx++) {
        nb_pt_idx = adj_list_[s][nb_idx];
        if (!visited[nb_pt_idx])
          nb_visited++;

        if (!visited_local[nb_pt_idx]) {
          visited[nb_pt_idx] = true;
          visited_local[nb_pt_idx] = true;

          // Fill in the adjacency matrix
          for (uint i=0; i < neighborhood[nb_pt_idx].size(); i++) {
            int node_idx2 = neighborhood[nb_pt_idx][i];
            adj_mat[node_idx*nodes_nb_ + node_idx2] = true;
            adj_mat[node_idx2*nodes_nb_ + node_idx] = true;
          }
          neighborhood[nb_pt_idx].push_back(node_idx);

          queue.push_back(nb_pt_idx);
        }
      }
    } // while queue not empty
  } // for each node


  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++) {
    valid_indices_[i] = true;
    adj_mat[i*nodes_nb_ + i] = true;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::getValidIndices(int* valid_indices) {
  for (uint i=0; i < nodes_nb_; i++) {
    if (valid_indices_[i])
      valid_indices[i] = 1;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADJACENCY ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::occupancyAdjacency(double* adj_mat, unsigned int neigh_nb) {
  pcl::PointCloud<PointT>::Ptr local_cloud(new pcl::PointCloud<PointT>);
  for (uint pt_idx=0; pt_idx < sampled_indices_.size(); pt_idx++) {
    local_cloud->points.push_back(pc_->points[sampled_indices_[pt_idx]]);
  }

  pcl::search::KdTree<PointT>::Ptr local_tree(new pcl::search::KdTree<PointT>);
  local_tree->setInputCloud(local_cloud);
  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;
  float occ_ratio;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    PointT pt = local_cloud->points[pt_idx];
    uint k_elts = neigh_nb < local_cloud->points.size() ? neigh_nb : local_cloud->points.size();
    local_tree->nearestKSearch(pt, k_elts, k_indices, k_sqr_distances);

    Eigen::Vector4f v1 = local_cloud->points[pt_idx].getVector4fMap ();
    adj_mat[nodes_nb_*pt_idx + pt_idx] = 1.;
    for (uint i=0; i < k_elts; i++) {
      if (k_sqr_distances[i] == 0.)
        continue;

      Eigen::Vector4f v2 = local_cloud->points[k_indices[i]].getVector4fMap ();
      occ_ratio = occupancy_ratio(v1, v2, lut_, gridsize_/2);
      if (occ_ratio > 0.0) {
        adj_mat[nodes_nb_*pt_idx + k_indices[i]] = 1.;
        adj_mat[nodes_nb_*k_indices[i] + pt_idx] = 1.;
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::fullConnectionAdjacency(double* adj_mat) {
  ScopeTime t("Adjacency matrix computation", debug_);
  for (uint index1=0; index1 < nodes_nb_; index1++) {
    for (uint index2=0; index2 < nodes_nb_; index2++) {
      adj_mat[index1*nodes_nb_ + index2] = 1.;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::correctAdjacencyForValidity(double* adj_mat) {
  for (uint index1=0; index1 < nodes_nb_; index1++) {
    for (uint index2=0; index2 < nodes_nb_; index2++) {
      if (!valid_indices_[index1] || !valid_indices_[index2])
        adj_mat[index1*nodes_nb_ + index2] = 0.;
    }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// EDGE FEATURES ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::coordsEdgeFeatures(double* edge_feats) {
  int index1, index2;

  for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
    index1 = sampled_indices_[pt1_idx];
    Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector3f v21 = v2 - v1;
      // v21.normalize();

      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 0] = v21(0);
      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 1] = v21(1);
      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 2] = v21(2);
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::rotZEdgeFeatures(double* edge_feats, float min_angle_z_normal) {
  int index1, index2;
  Eigen::Vector3f z(0., 0., 1.);
  z.normalize();

  for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
    index1 = sampled_indices_[pt1_idx];
    Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();
    Eigen::Vector3f n1 = pc_->points[index1].getNormalVector3fMap();
    n1.normalize();

    if (acos(fabs(n1.dot(z))) < min_angle_z_normal*M_PI/180.) {
      valid_indices_[pt1_idx] = false;
      continue;
    }


    Eigen::Vector3f n_axis;
    n_axis(0) = n1(0);
    n_axis(1) = n1(1);
    n_axis(2) = 0.;
    n_axis.normalize();
    Eigen::Vector3f w = n_axis.cross(z);

    Eigen::Matrix3f lrf;

    lrf.row(0) << n1;
    lrf.row(1) << w;
    lrf.row(2) << z;

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector3f v21 = v2 - v1;
      Eigen::Vector3f local_coords = lrf * v21;
      // v21.normalize();

      if (std::isnan(local_coords(0)) || std::isnan(local_coords(1)) || std::isnan(local_coords(2))) {
        std::cout << local_coords << "\n----\n"
                  << lrf << "\n---\n"
                  << n1 << "\n---\n"
                  << n_axis << "\n---\n"
                  << w << "\n---\n"
                  << z << "\n***\n"
                  << std::endl;
      }

      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 0] = local_coords(0);
      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 1] = local_coords(1);
      edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 2] = local_coords(2);
    }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// NODE FEATURES ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::lEsfNodeFeatures(double** result, unsigned int feat_nb) {
  uint p1_idx, p2_idx, rdn_weight;
  uint pair_nb=0;
  uint max_pair_nb;
  const uint sample_pair_nb = feat_nb;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  Eigen::Vector4f v1, v2, n1, n2, v12;
  double dist, a12, a12n1, a12n2, occ_r, z;
  double feat_min = -0.5;
  double feat_max =  0.5;
  // double max_dist = 0.;

  // // if (params_.mesh) {
  // //   for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
  // //     v1 = pc_->points[nodes_elts_[pt_idx][0]].getVector4fMap ();
  // //     for (uint i=nodes_elts_[pt_idx].size()/2; i < nodes_elts_[pt_idx].size(); i++) {
  // //       v2 = pc_->points[nodes_elts_[pt_idx][i]].getVector4fMap ();
  // //       v12 = v1-v2;
  // //       double dist = 2.*v12.norm();
  // //       if (dist > max_dist)
  // //         max_dist = dist;
  // //     }
  // //   }
  // // } else
  // //   max_dist = 2*params_.neigh_size;
  // max_dist = gridsize_;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    k_indices = nodes_elts_[pt_idx];
    //tree_->radiusSearch(pc_->points[sampled_indices_[pt_idx]], params_.neigh_size, k_indices, k_sqr_distances);

    pair_nb = 0;
    max_pair_nb = k_indices.size() * (k_indices.size() - 1) / 2;

    if (max_pair_nb < sample_pair_nb / 2) {
      if (debug_)
        std::cout << "Node " << pt_idx << " max pair nb: " << max_pair_nb << std::endl;

      valid_indices_[pt_idx] = false;
    }

    for (uint index1=0; index1 < k_indices.size(); index1++) {
      for (uint index2=index1+1; index2 < k_indices.size(); index2++) {
        if (max_pair_nb < 10*sample_pair_nb) {
          rdn_weight = rand() % max_pair_nb;
          if (rdn_weight > sample_pair_nb)
            continue;

          p1_idx = k_indices[index1];
          p2_idx = k_indices[index2];
        } else {
          int max_sampling = std::min(static_cast<int>(k_indices.size()), 500);
          rdn_weight = rand() % max_sampling;
          p1_idx = k_indices[rdn_weight];
          // rdn_weight = rand() % k_indices.size();
          // p2_idx = k_indices[rdn_weight];
          p2_idx = rand() % pc_->points.size();
        }

        if (std::isnan(pc_->points[p1_idx].normal_x) || std::isnan(pc_->points[p2_idx].normal_x))
          continue;

        // Get the vectors
        v1 = pc_->points[p1_idx].getVector4fMap ();
        v2 = pc_->points[p2_idx].getVector4fMap ();
        n1 = pc_->points[p1_idx].getNormalVector4fMap ();
        n2 = pc_->points[p2_idx].getNormalVector4fMap ();

        v12 = v1 - v2;


        // Feature computation
        // dist = scale_ * v12.norm() / (gridsize_) - 0.5;
        // z = scale_ * v12(2) / gridsize_ - 0.5;
        dist = v12.norm() / (gridsize_) - 0.5;
        z = v12(2) / gridsize_ - 0.5;
        v12.normalize();
        occ_r = occupancy_ratio(v1, v2, lut_, gridsize_/2) - 0.5;
        a12 = n1.dot(n2)/2.;
        a12n1 = fabs(v12.dot(n1)) - 0.5;
        a12n2 = fabs(v12.dot(n2)) - 0.5;


        // Saturate the features
        dist = std::min(std::max(dist, feat_min), feat_max);
        a12 = std::min(std::max(a12, feat_min), feat_max);
        a12n1 = std::min(std::max(a12n1, feat_min), feat_max);
        a12n2 = std::min(std::max(a12n2, feat_min), feat_max);

        if (std::isnan(dist) || std::isnan(a12) || std::isnan(a12n1) ||
            std::isnan(a12n2) || std::isnan(occ_r) || std::isnan(z))
          continue;


        // Fill in the matrix
        result[pt_idx][6*pair_nb + 0] = dist;
        result[pt_idx][6*pair_nb + 1] = occ_r;
        result[pt_idx][6*pair_nb + 2] = a12;
        result[pt_idx][6*pair_nb + 3] = a12n1;
        result[pt_idx][6*pair_nb + 4] = a12n2;
        result[pt_idx][6*pair_nb + 5] = z;

        pair_nb += 1;

        if (pair_nb >= sample_pair_nb) {
          // Break out of the two loops
          index1 = k_indices.size();
          index2 = k_indices.size();
        }
      }
    }

    // // Normalize
    // for (uint i=0; i<64; i++) {
    //   result[pt_idx][i] /= pair_nb + 1e-6;
    //   // result[pt_idx][i] -= 0.5;
    // }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::coordsSetNodeFeatures(double** result, unsigned int feat_nb) {
  PointT p;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  const uint sample_neigh_nb = feat_nb;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    k_indices = nodes_elts_[pt_idx];
    // tree_->radiusSearch(pc_->points[sampled_indices_[pt_idx]], params_.neigh_size, k_indices, k_sqr_distances);

    for (uint index1=0; index1 < k_indices.size(); index1++) {
      if (index1 >= sample_neigh_nb)
        break;

      p = pc_->points[k_indices[index1]];

      // Fill in the matrix
      result[pt_idx][3*index1 + 0] = p.x * 2. / gridsize_;
      result[pt_idx][3*index1 + 1] = p.y * 2. / gridsize_;
      result[pt_idx][3*index1 + 2] = p.z * 2. / gridsize_;
    }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::viz(double* adj_mat, bool viz_small_spheres) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1., "coords", 0);

  for (uint i=0; i<sampled_indices_.size(); i++) {
    int idx = sampled_indices_[i];

    PointT p1 = pc_->points[idx];
    PointT p2_n = pc_->points[idx];
    p2_n.x -= 5*p2_n.normal_x;
    p2_n.y -= 5*p2_n.normal_y;
    p2_n.z -= 5*p2_n.normal_z;

    PointT p2_u = pc_->points[idx];
    p2_u.x -= 5*p2_u.normal_x;
    p2_u.y -= 5*p2_u.normal_y;

    PointT p2_z = pc_->points[idx];
    p2_z.z += 5*1.;

    // viewer->addArrow (p2_n, p1, 1., 1., 1., false, "arrow_n" + std::to_string(i));
    // viewer->addArrow (p2_u, p1, 0., 1., 1., false, "arrow_u" + std::to_string(i));
    // viewer->addArrow (p2_z, p1, 1., 0., 1., false, "arrow_z" + std::to_string(i));

    if (viz_small_spheres)
      viewer->addSphere<PointT>(pc_->points[idx], 0.05, 1., 0., 0., "sphere_" +std::to_string(idx));
    // else
    //   viewer->addSphere<PointT>(pc_->points[idx], params_.neigh_size, 1., 0., 0., "sphere_" +std::to_string(idx));

    for (uint i2=0; i2<nodes_nb_; i2++) {
      if (adj_mat[nodes_nb_*i + i2] > 0.) {
        int idx2 = sampled_indices_[i2];
        if (idx != idx2)
          viewer->addLine<PointT>(pc_->points[idx], pc_->points[idx2], 1., 1., 0., "line_" +std::to_string(idx)+std::to_string(idx2));
      }
    }
  }

  viewer->addPointCloud<PointT> (pc_, "cloud");
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}
