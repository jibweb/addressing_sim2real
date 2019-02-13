#include <random>
#include <math.h>
#include <unordered_map>
#include <utility>
// #include <time>

#include <boost/functional/hash.hpp>
#include <pcl/conversions.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <igl/writePLY.h>

#include "graph_construction.h"
#include "bresenham.cpp"

typedef std::pair<uint, uint> Edge;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// GENERAL ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float triangle_area(Eigen::Vector4f& p1, Eigen::Vector4f& p2, Eigen::Vector4f& p3) {
  float a,b,c,s;

  // Get the area of the triangle
  Eigen::Vector4f v21 (p2 - p1);
  Eigen::Vector4f v31 (p3 - p1);
  Eigen::Vector4f v23 (p2 - p3);
  a = v21.norm (); b = v31.norm (); c = v23.norm (); s = (a+b+c) * 0.5f + 1e-6;

  return sqrt(s * (s-a) * (s-b) * (s-c));
}


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
void GraphConstructor::initializeMesh(float min_angle_z_normal, double* adj_mat, float neigh_size) {

  ScopeTime t("Initialization (MeshGraphConstructor)", debug_);

  // Read the point cloud
  if (pcl::io::loadPLYFile(filename_.c_str(), *mesh_) == -1) {
    PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
    return;
  }

  pcl::fromPCLPointCloud2(mesh_->cloud, *pc_);

  // std::string pc_filename = filename_.substr(0, filename_.size() - 4) + ".pcd";
  // // Read the point cloud
  // if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (pc_filename.c_str(), *pc_) == -1) {
  //   PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
  //   return;
  // }


  if (debug_) {
    std::cout << "PolygonMesh: " << mesh_->polygons.size() << " triangles" << std::endl;
    std::cout << "PC size: " << pc_->points.size() << std::endl;
  }

  // Data augmentation
  scale_points_unit_sphere (*pc_, 1.);
  // scale_points_unit_sphere (*pc_, gridsize_/2, centroid);
  // params_.neigh_size = params_.neigh_size * gridsize_/2;
  // augment_data(pc_, params_);

  // Initialize the tree
  tree_->setInputCloud (pc_);

  nodes_elts_.resize(nodes_nb_);
  for (uint i=0; i < nodes_elts_.size(); i++)
    nodes_elts_[i].reserve(100);

  // TODO Remove nodes with a wrong angle_z_normal

  // Initialize the valid indices
  valid_indices_.resize(nodes_nb_);



  // --- Edge connectivity ----------------------------------------------------
  std::unordered_map<Edge,std::vector<uint>,boost::hash<Edge> > edge_to_triangle;
  // std::unordered_map<Edge, std::vector<uint> > edge_to_triangle;

  for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
    for(uint edge_idx=0; edge_idx<3; edge_idx++) {
      uint idx1 = mesh_->polygons[tri_idx].vertices[edge_idx];
      uint idx2 = mesh_->polygons[tri_idx].vertices[(edge_idx+1)%3];

      if (idx1 == idx2)
        std::cout << "Two vertices of a triangle have the same index aka weird duplicates that are part of a triangle somehow" << std::endl;

      if (idx1 > idx2) {
        Edge edge_id(idx2,idx1);
        edge_to_triangle[edge_id].push_back(tri_idx);
      } else {
        Edge edge_id(idx1,idx2);
        edge_to_triangle[edge_id].push_back(tri_idx);
      }
    }
  }

  // --- Face area ------------------------------------------------------------
  std::vector<float> face_area;
  float total_area = 0.;
  face_area.resize(mesh_->polygons.size());
  for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
    Eigen::Vector4f p1 = pc_->points[mesh_->polygons[tri_idx].vertices[0]].getVector4fMap ();
    Eigen::Vector4f p2 = pc_->points[mesh_->polygons[tri_idx].vertices[1]].getVector4fMap ();
    Eigen::Vector4f p3 = pc_->points[mesh_->polygons[tri_idx].vertices[2]].getVector4fMap ();
    face_area[tri_idx] = triangle_area(p1, p2, p3);
    total_area += face_area[tri_idx];
  }

  float target_area = neigh_size; //total_area * neigh_size; // static_cast<float>(neigh_nb) / mesh_->polygons.size();

  if (debug_)
    std::cout << "Target area: " << target_area << " / Total area: " << total_area  << std::endl; //<< " / Proportion: " << neigh_size << std::endl; //static_cast<float>(neigh_nb) / mesh_->polygons.size() << std::endl;

  // --- BFS surface sampling (area dependent) --------------------------------
  // --- Global setup for the sampling procedure ------------------------------
  srand (static_cast<unsigned int> (time (0)));

  // --- Do a BFS per node in the graph ---------------------------------------
  std::vector<int> node_association(mesh_->polygons.size(), -1);

  for (uint node_idx=0; node_idx < nodes_nb_; node_idx++) {
    // --- Select a node and enqueue it ---------------------------------------
    float rdn_weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/total_area));
    uint sampled_idx = mesh_->polygons.size() + 1;

    for (uint tri_idx=0; tri_idx<face_area.size(); tri_idx++) {
      rdn_weight -= face_area[tri_idx];

      if (rdn_weight <= 0.) {
        sampled_idx = tri_idx;
        break;
      }
    }

    // Failed to sample a point
    if (sampled_idx == (mesh_->polygons.size() + 1))
      break;

    sampled_indices_.push_back(sampled_idx);

    // --- Setup for BFS ------------------------------------------------------
    std::deque<int> queue;
    queue.push_back(sampled_idx);

    float sampled_area = 0.; //face_area[sampled_idx];

    std::vector<bool> visited(mesh_->polygons.size(), false);
    visited[sampled_idx] = true;

    // --- BFS over the graph to extract the neighborhood ---------------------
    while(!queue.empty() && sampled_area < target_area) {
      // Dequeue a face
      int s = queue.front();
      queue.pop_front();

      // Update the areas
      nodes_elts_[node_idx].push_back(s);
      node_association[s] = node_idx;
      if (face_area[s] > (target_area - sampled_area)) {
        sampled_area = target_area;
        total_area -= target_area - sampled_area;
        face_area[s] -= target_area - sampled_area;
      } else {
        sampled_area += face_area[s];
        total_area -= face_area[s];
        face_area[s] = 0.;
      }

      // For each edge, find the biggest unvisited neighbor and visit all of them
      for (uint edge_idx=0; edge_idx<3; edge_idx++) {
        uint idx1 = mesh_->polygons[s].vertices[edge_idx];
        uint idx2 = mesh_->polygons[s].vertices[(edge_idx+1)%3];

        if (idx1 > idx2) {
          uint tmp = idx1;
          idx1 = idx2;
          idx2 = tmp;
        }

        Edge edge_id(idx1,idx2);
        int neigh_tri_idx = -1;
        float max_surface_area = -1.;
        for (uint i=0; i<edge_to_triangle[edge_id].size(); i++) {

          if (node_association[edge_to_triangle[edge_id][i]] != -1) {
            uint node_idx2 = node_association[edge_to_triangle[edge_id][i]];
            adj_mat[node_idx*nodes_nb_ + node_idx2] = true;
            adj_mat[node_idx2*nodes_nb_ + node_idx] = true;
          }

          if (visited[edge_to_triangle[edge_id][i]]) {
            continue;
          } else {
            visited[edge_to_triangle[edge_id][i]] = true;
          }
          // This is not the triangle you're looking for
          if (edge_to_triangle[edge_id][i] == s)
            continue;

          // If we're here, it is one of the unvisited neighboring triangles
          if (face_area[edge_to_triangle[edge_id][i]] > max_surface_area) {
            neigh_tri_idx = edge_to_triangle[edge_id][i];
            max_surface_area = face_area[edge_to_triangle[edge_id][i]];
          }
        } // --loop over edge neighbors

        // If a unvisited triangle was found for that edge, enqueue it
        if (neigh_tri_idx != -1 && max_surface_area > 0.) {
          queue.push_back(neigh_tri_idx);
        }

      } // --loop over triangle edges
    } // while queue not empty

    if (debug_)
      std::cout << node_idx << " Sampled idx " << sampled_idx << " (Contains " << nodes_elts_[node_idx].size() << " faces)" << std::endl;
  } // for each node

  for (auto area : face_area)
    if (area < 0.)
      std::cout<< "Areas" << area << std::endl;


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
void GraphConstructor::sphNodeFeatures(double** result, uint image_size, uint r_sdiv, uint p_sdiv) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXd V_uv;

  {
    ScopeTime t("Subset extraction computation", debug_);
    // --- SUBSET EXTRACTION --------------------------------------------------
    // Extract the proper vertex subset corresponding to our face subset
    std::set<uint> vertex_subset;
    for (uint tri_idx=0; tri_idx < nodes_elts_[0].size(); tri_idx++) {
      uint face_idx = nodes_elts_[0][tri_idx];
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[0]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[1]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[2]);
    }

    // Re-map vertices of the subset to 0-vertices_nb to re-index the triangles properly
    std::unordered_map<uint, uint> reverse_vertex_idx;
    uint new_idx = 0;
    for (auto vertex_idx : vertex_subset) {
      reverse_vertex_idx[vertex_idx] = new_idx;
      new_idx++;
    }

    // --- LibIGL setup ---------------------------------------------------------
    V.resize(vertex_subset.size(), 3);
    F.resize(nodes_elts_[0].size(), 3);

    if (debug_)
      std::cout << "Node size (faces): " << nodes_elts_[0].size() << " | "
                << "Vertex subset size: " << vertex_subset.size() << std::endl;

    // Fill in V
    uint v_idx=0;
    for (auto vertex_idx : vertex_subset) {
      V(v_idx,0) = pc_->points[vertex_idx].x;
      V(v_idx,1) = pc_->points[vertex_idx].y;
      V(v_idx,2) = pc_->points[vertex_idx].z;
      v_idx++;
    }

    // Fill in F
    for (uint loop_idx=0; loop_idx < nodes_elts_[0].size(); loop_idx++) {
      uint face_idx = nodes_elts_[0][loop_idx];
      F(loop_idx, 0) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[0]];
      F(loop_idx, 1) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[1]];
      F(loop_idx, 2) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[2]];
    }
  }

  std::cout << "LSCM begin" << std::endl;
  {
    ScopeTime t("LSCM subcomputation", debug_);

    // Fix two points on the boundary
    igl::writePLY("./extracted_subset.ply", V, F);
    Eigen::VectorXi bnd,b(2,1);
    igl::boundary_loop(F,bnd);
    std::cout << "A1" << std::endl;
    std::cout << "bnd.size() " << bnd.size() << std::endl;
    // std::cout << "bnd " << bnd << std::endl;
    b(0) = bnd(0);
    int idx_b1 = round(bnd.size()/2);
    std::cout << "A2" << std::endl;
    b(1) = bnd(idx_b1);
    Eigen::MatrixXd bc(2,2);
    bc<<0,0,1,0;
    std::cout << "A3" << std::endl;

    // LSCM parametrization
    igl::lscm(V,F,b,bc,V_uv);
  }

  // Scale the uv
  V_uv *= 5;



  std::cout << "Rasterizer begin" << std::endl;
  {
    ScopeTime t("Rasterizer computation", debug_);

    double max_u = -50.;
    double min_u = 50.;
    double max_v = -50.;
    double min_v = 50.;
    for (uint i=0; i<V_uv.rows(); i++) {
      if (V_uv(i,0) > max_u) {
        max_u = V_uv(i,0);
      }
      if (V_uv(i,0) < min_u) {
        min_u = V_uv(i,0);
      }
      if (V_uv(i,1) > max_v) {
        max_v = V_uv(i,1);
      }
      if (V_uv(i,1) < min_v) {
        min_v = V_uv(i,1);
      }
    }

    // --- Finding out the center triangle of the map -----------------------
    uint center_tri_idx = 0;
    double u_center_pt = (max_u + min_u) / 2;
    double v_center_pt = (max_v + min_v) / 2;
    Eigen::Vector3d vcenter;

    for (uint face_idx=0; face_idx<F.rows(); face_idx++) {
      double tri_max_u = std::max(V_uv(F(face_idx, 0), 0), std::max(V_uv(F(face_idx, 1), 0), V_uv(F(face_idx, 2), 0)));
      double tri_min_u = std::min(V_uv(F(face_idx, 0), 0), std::min(V_uv(F(face_idx, 1), 0), V_uv(F(face_idx, 2), 0)));
      if ((u_center_pt > tri_max_u) || (u_center_pt < tri_min_u))
        continue;

      double tri_max_v = std::max(V_uv(F(face_idx, 0), 1), std::max(V_uv(F(face_idx, 1), 1), V_uv(F(face_idx, 2), 1)));
      double tri_min_v = std::min(V_uv(F(face_idx, 0), 1), std::min(V_uv(F(face_idx, 1), 1), V_uv(F(face_idx, 2), 1)));
      if ((v_center_pt > tri_max_v) || (v_center_pt < tri_min_v))
        continue;

      // Center point is within the bounding box. Now check if it's in the triangle
      double w0 = edgeFunction(V_uv(F(face_idx,1), 0), V_uv(F(face_idx,1), 1),
                               V_uv(F(face_idx,2), 0), V_uv(F(face_idx,2), 1),
                               u_center_pt, v_center_pt);
      double w1 = edgeFunction(V_uv(F(face_idx,2), 0), V_uv(F(face_idx,2), 1),
                               V_uv(F(face_idx,0), 0), V_uv(F(face_idx,0), 1),
                               u_center_pt, v_center_pt);
      double w2 = edgeFunction(V_uv(F(face_idx,0), 0), V_uv(F(face_idx,0), 1),
                               V_uv(F(face_idx,1), 0), V_uv(F(face_idx,1), 1),
                               u_center_pt, v_center_pt);

      if ((w0 < 0. && w1 < 0. && w2 < 0.) || (w0 > 0. && w1 > 0. && w2 > 0.)) {
        center_tri_idx = face_idx;
        double area = edgeFunction(V_uv(F(face_idx,0), 0), V_uv(F(face_idx,0), 1),
                                   V_uv(F(face_idx,1), 0), V_uv(F(face_idx,1), 1),
                                   V_uv(F(face_idx,2), 0), V_uv(F(face_idx,2), 1));
        w0 /= area;
        w1 /= area;
        w2 /= area;

        vcenter = fabs(w0)*V.row(F(face_idx,0)) + fabs(w1)*V.row(F(face_idx,1)) + fabs(w2)*V.row(F(face_idx,2));
        break;
      }
    }


    // --- Vertices features computation ------------------------------------
    // TODO change to correct center face once I sample per triangle
    Eigen::Vector3d vcenter0 = V.row(F(center_tri_idx, 0));
    Eigen::Vector3d vcenter1 = V.row(F(center_tri_idx, 1));
    Eigen::Vector3d vcenter2 = V.row(F(center_tri_idx, 2));

    Eigen::Vector3d vcenter01 = vcenter0 - vcenter1;
    Eigen::Vector3d vcenter02 = vcenter0 - vcenter2;

    Eigen::Vector3d n_centertri = vcenter01.cross(vcenter02);
    n_centertri.normalize();

    // Get coordinates centered around the sample point
    Eigen::MatrixXd V_centered = V;
    V_centered.rowwise() -= vcenter.transpose();

    // Compute the euclidean distance
    Eigen::VectorXd Ved = V_centered.rowwise().lpNorm<2>();

    // Compute the distance to the triangle plane
    Eigen::VectorXd Vpd;
    Vpd.resize(V.rows());
    for (uint i=0; i<Vpd.rows(); i++) {
      Vpd(i) = n_centertri.dot(V_centered.row(i));
    }

    // --- Rasterization ----------------------------------------------------
    Eigen::MatrixXd res_image_0 = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);
    Eigen::MatrixXd res_image_1 = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);

    for (uint face_idx=0; face_idx<F.rows(); face_idx++) {
      int min_x = image_size + 1;
      int max_x = -1;
      std::vector<int> vx(3);
      std::vector<int> vy(3);

      // Get the range in x coordinates
      for (uint i=0; i<3; i++) {
        vx[i] = image_size * (V_uv(F(face_idx, i), 0) - min_u) / (max_u - min_u);
        vy[i] = image_size * (V_uv(F(face_idx, i), 1) - min_v) / (max_v - min_v);

        if (vx[i] < min_x)
          min_x = vx[i];

        if (vx[i] > max_x)
          max_x = vx[i];
      }

      // Get the range of y for each x in range of this triangle
      // aka fully define the area where the triangle needs to be drawn
      std::vector<int> max_y(max_x-min_x+1, -1);
      std::vector<int> min_y(max_x-min_x+1, image_size + 1);

      for(uint i=0; i<3; i++) {
        bresenham_line(vx[i], vy[i], vx[(i+1)%3], vy[(i+1)%3], min_y, max_y, min_x);
      }

      // Once we have the boundaries of the triangles, draw it !
      float tri_area = abs((vx[2] - vx[0])*(vy[1] - vy[0]) - (vy[2] - vy[0])*(vx[1] - vx[0])); //Twice the area but who cares

      for (uint i=0; i<max_y.size(); i++) {
        // Compute the barycentric coordinates and the step update
        float w0 = (min_x + static_cast<int>(i) - vx[1]) * (vy[2] - vy[1]) - (min_y[i] - vy[1]) * (vx[2] - vx[1]);
        float w1 = (min_x + static_cast<int>(i) - vx[2]) * (vy[0] - vy[2]) - (min_y[i] - vy[2]) * (vx[0] - vx[2]);
        float w2 = (min_x + static_cast<int>(i) - vx[0]) * (vy[1] - vy[0]) - (min_y[i] - vy[0]) * (vx[1] - vx[0]);

        w0 /= tri_area;
        w1 /= tri_area;
        w2 /= tri_area;

        float w0_stepy = -(vx[2] - vx[1]) / tri_area;
        float w1_stepy = -(vx[0] - vx[2]) / tri_area;
        float w2_stepy = -(vx[1] - vx[0]) / tri_area;

        for (uint j=min_y[i]; j<max_y[i]; j++) {
          res_image_0((min_x + i), j) = fabs(w0)*Ved(F(face_idx, 0)) + fabs(w1)*Ved(F(face_idx, 1)) + fabs(w2)*Ved(F(face_idx, 2));
          res_image_1((min_x + i), j) = fabs(w0)*Vpd(F(face_idx, 0)) + fabs(w1)*Vpd(F(face_idx, 1)) + fabs(w2)*Vpd(F(face_idx, 2));

          w0 += w0_stepy;
          w1 += w1_stepy;
          w2 += w2_stepy;
        }
      }
    } // for loop on faces


    // --- Extract polar representation of our feature map ------------------
    uint half_image_size = image_size/2;
    uint x_px, y_px;
    r_sdiv++;
    for (uint r=1; r<r_sdiv; r++) {
      for (uint p=0; p<p_sdiv; p++) {
        x_px = static_cast<uint>(half_image_size*r*cos(2.*M_PI*p / p_sdiv)/r_sdiv + half_image_size);
        y_px = static_cast<uint>(half_image_size*r*sin(2.*M_PI*p / p_sdiv)/r_sdiv + half_image_size);
        result[0][(r-1)*p_sdiv*2 + p*2 + 0] = res_image_0(x_px, y_px);
        result[0][(r-1)*p_sdiv*2 + p*2 + 1] = res_image_1(x_px, y_px);

        // std::cout << "[" << x_px << ", " << y_px << "]," << std::endl;
      }
    }
  } // Rasterizer scope
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::viz(double* adj_mat, bool viz_small_spheres) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  {
    ScopeTime t("PC viz computation", debug_);
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
  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::vizMesh(double* adj_mat, bool viz_small_spheres) {
  // --- Viz ------------------------------------------------------------------

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  {
    ScopeTime t("Mesh viz computation", debug_);
    viewer->setBackgroundColor (0, 0, 0);
    // viewer->addCoordinateSystem (1., "coords", 0);

    for (uint i=0; i<sampled_indices_.size(); i++) {
      uint tri_idx = sampled_indices_[i];
      Eigen::Vector4f p1 = pc_->points[mesh_->polygons[tri_idx].vertices[0]].getVector4fMap ();
      Eigen::Vector4f p2 = pc_->points[mesh_->polygons[tri_idx].vertices[1]].getVector4fMap ();
      Eigen::Vector4f p3 = pc_->points[mesh_->polygons[tri_idx].vertices[2]].getVector4fMap ();

      p1 += p2;
      p1 += p3;
      p1 /= 3;

      PointT p;
      p.x = p1(0);
      p.y = p1(1);
      p.z = p1(2);

      if (i == 0)
        viewer->addSphere<PointT>(p, 0.015, 1., 1., 0., "sphere_zero");
      else
        viewer->addSphere<PointT>(p, 0.01, 1., 0., 0., "sphere_" +std::to_string(tri_idx));

      for (uint i2=0; i2<nodes_nb_; i2++) {
        if (adj_mat[nodes_nb_*i + i2] > 0.) {
          int idx2 = sampled_indices_[i2];
          if (tri_idx != idx2) {
            Eigen::Vector4f p2_1 = pc_->points[mesh_->polygons[idx2].vertices[0]].getVector4fMap ();
            Eigen::Vector4f p2_2 = pc_->points[mesh_->polygons[idx2].vertices[1]].getVector4fMap ();
            Eigen::Vector4f p2_3 = pc_->points[mesh_->polygons[idx2].vertices[2]].getVector4fMap ();

            p2_1 += p2_2;
            p2_1 += p2_3;
            p2_1 /= 3;

            PointT p2;
            p2.x = p2_1(0);
            p2.y = p2_1(1);
            p2.z = p2_1(2);
            viewer->addLine<PointT>(p, p2, 0., 0., 1., "line_" +std::to_string(tri_idx)+std::to_string(idx2));
          }
        }
      }
    }


    std::vector<uint> r, g, b;
    r.resize(sampled_indices_.size());
    g.resize(sampled_indices_.size());
    b.resize(sampled_indices_.size());
    for (uint i =0; i < sampled_indices_.size(); i++){
      r[i] = rand() % 255;
      b[i] = rand() % 255;
      g[i] = rand() % 255;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
      pcl::PointXYZRGB p;
      p.x = pc_->points[pt_idx].x;
      p.y = pc_->points[pt_idx].y;
      p.z = pc_->points[pt_idx].z;

      p.r = 230;
      p.g = 230;
      p.b = 230;

      viz_cloud->points.push_back(p);
    }

    for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {
      for (uint pt_idx=0; pt_idx<nodes_elts_[node_idx].size(); pt_idx++) {
        uint tri_idx = nodes_elts_[node_idx][pt_idx];
        for (uint i=0; i<3; i++) {
          viz_cloud->points[mesh_->polygons[tri_idx].vertices[i]].r = r[node_idx];
          viz_cloud->points[mesh_->polygons[tri_idx].vertices[i]].g = g[node_idx];
          viz_cloud->points[mesh_->polygons[tri_idx].vertices[i]].b = b[node_idx];
        }
      }
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(viz_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (viz_cloud, rgb, "cloud");


    // Viz resized mesh
    pcl::PolygonMesh::Ptr mesh_2(new pcl::PolygonMesh);
    pcl::PCLPointCloud2 point_cloud2;
    pcl::toPCLPointCloud2(*pc_, point_cloud2);

    mesh_2->cloud = point_cloud2;
    mesh_2->polygons = mesh_->polygons;
    viewer->addPolygonMesh (*mesh_2);

  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }

}

