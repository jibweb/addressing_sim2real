#include <random>
#include <math.h>
#include <unordered_map>
#include <utility>
// #include <time>

#include <boost/functional/hash.hpp>
#include <pcl/common/pca.h>
#include <pcl/conversions.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <igl/writePLY.h>

#include "graph_construction.h"
#include "mesh_utils.cpp"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// GENERAL ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::pair<uint, uint> Edge;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::initializeMesh(float min_angle_z_normal, double* adj_mat, float neigh_size) {

  if (!debug_)
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  // TODO : probably should be a param. Weird condition for meshes with triangles of variable areas
  uint min_node_size = 120;

  ScopeTime t("Initialization (MeshGraphConstructor)", debug_);
  // boost::posix_time::ptime start_time_ = boost::posix_time::microsec_clock::local_time ();
  // Read the point cloud
  if (pcl::io::loadPLYFile(filename_.c_str(), *mesh_) == -1) {
    PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
    return;
  }

  // std::cout << "A0 " << (static_cast<double> (((boost::posix_time::microsec_clock::local_time () - start_time_).total_milliseconds ()))) << std::endl;

  pcl::fromPCLPointCloud2(mesh_->cloud, *pc_);
  tree_->setInputCloud (pc_);

  if (debug_) {
    std::cout << "PolygonMesh: " << mesh_->polygons.size() << " triangles" << std::endl;
    std::cout << "PC size: " << pc_->points.size() << std::endl;
  }

  // Data augmentation
  scale_points_unit_sphere (*pc_, 1.);
  // scale_points_unit_sphere (*pc_, gridsize_/2, centroid);
  // params_.neigh_size = params_.neigh_size * gridsize_/2;
  // augment_data(pc_, params_);

  nodes_elts_.resize(nodes_nb_);
  for (uint i=0; i < nodes_elts_.size(); i++)
    nodes_elts_[i].reserve(min_node_size);

  // TODO Remove nodes with a wrong angle_z_normal

  // Initialize the valid indices
  valid_indices_.resize(nodes_nb_);

  // std::cout << "A " << (static_cast<double> (((boost::posix_time::microsec_clock::local_time () - start_time_).total_milliseconds ()))) << std::endl;



  // --- Edge connectivity ----------------------------------------------------
  std::unordered_map<Edge, std::array<int, 2>, boost::hash<Edge> > edge_to_triangle;
  edge_to_triangle.reserve(3*mesh_->polygons.size());

  for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
    for(uint edge_idx=0; edge_idx<3; edge_idx++) {
      uint idx1 = mesh_->polygons[tri_idx].vertices[edge_idx];
      uint idx2 = mesh_->polygons[tri_idx].vertices[(edge_idx+1)%3];

      if (idx1 > idx2) {
        uint tmp = idx1;
        idx1 = idx2;
        idx2 = tmp;
      }

      Edge edge_id(idx2,idx1);

      auto arr = edge_to_triangle.find(edge_id);

      if (arr != edge_to_triangle.end()) {
        // Edge exists already
        arr->second[1] = tri_idx;
      } else {
        // Edge doesn't exist yet
        edge_to_triangle[edge_id][0] = tri_idx;
        edge_to_triangle[edge_id][1] = -1;
      }
    }
  }

  // std::cout << "B1 " << (static_cast<double> (((boost::posix_time::microsec_clock::local_time () - start_time_).total_milliseconds ()))) << std::endl;


  std::vector<std::vector<uint> > triangle_neighbors(mesh_->polygons.size());
  for (uint i=0; i<triangle_neighbors.size(); i++)
    triangle_neighbors[i].reserve(3);

  for(auto& it : edge_to_triangle) {
    // TODO slightly shitty way of resolving non manifold meshes
    // if (it.second.size() >= 2) {
    if (it.second[1] != -1) {
      triangle_neighbors[it.second[0]].push_back(it.second[1]);
      triangle_neighbors[it.second[1]].push_back(it.second[0]);
    }
    // }
    // else {
    //   for (uint tri1=0; tri1 < it.second.size(); tri1++) {
    //     for (uint tri2=0; tri2 < it.second.size(); tri2++) {
    //       if (tri1 == tri2)
    //         continue;
    //       triangle_neighbors[it.second[tri1]].push_back(it.second[tri2]);
    //       triangle_neighbors[it.second[tri2]].push_back(it.second[tri1]);
    //     }
    //   }
    // }
  }

  // std::cout << "B " << (static_cast<double> (((boost::posix_time::microsec_clock::local_time () - start_time_).total_milliseconds ()))) << std::endl;


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

  std::vector<float> samplable_face_area = face_area;

  float target_area = neigh_size;

  if (debug_)
    std::cout << "Target area: " << target_area << " / Total area: " << total_area  << std::endl; //<< " / Proportion: " << neigh_size << std::endl; //static_cast<float>(neigh_nb) / mesh_->polygons.size() << std::endl;

  // --- BFS surface sampling (area dependent) --------------------------------
  // --- Global setup for the sampling procedure ------------------------------
  srand (static_cast<unsigned int> (time (0)));

  // --- Do a BFS per node in the graph ---------------------------------------
  // std::vector<std::vector<int> > node_association(mesh_->polygons.size());
  node_face_association_.resize(mesh_->polygons.size());
  for (uint i=0; i<node_face_association_.size(); i++)
    node_face_association_[i].reserve(4);

  for (uint node_idx=0; node_idx < nodes_nb_; node_idx++) {
    // --- Select a node and enqueue it ---------------------------------------
    if (total_area <= 0.f)
      break;

    float rdn_weight = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/total_area));
    uint sampled_idx = mesh_->polygons.size() + 1;

    for (uint tri_idx=0; tri_idx<samplable_face_area.size(); tri_idx++) {
      rdn_weight -= samplable_face_area[tri_idx];

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

    float sampled_area = 0.;

    std::vector<bool> visited(mesh_->polygons.size(), false);
    visited[sampled_idx] = true;

    // --- BFS over the graph to extract the neighborhood ---------------------
    while(!queue.empty() && sampled_area < target_area) {
      // Dequeue a face
      int s = queue.front();
      queue.pop_front();

      // Update the areas
      nodes_elts_[node_idx].push_back(s);
      node_face_association_[s].push_back(node_idx);

      if (samplable_face_area[s] > (target_area - sampled_area)) {
        total_area = std::max(0.f, total_area  - target_area + sampled_area);
        samplable_face_area[s] = samplable_face_area[s] - (target_area - sampled_area);
      } else {
        total_area = std::max(0.f, total_area  - samplable_face_area[s]);
        samplable_face_area[s] = 0.f;
      }

      sampled_area += face_area[s];


      // For each edge, find the unvisited neighbor and visit all of them
      for (uint neigh_idx=0; neigh_idx<triangle_neighbors[s].size(); neigh_idx++) {
        uint neigh_tri = triangle_neighbors[s][neigh_idx];

        if (visited[neigh_tri])
          continue;

        visited[neigh_tri] = true;

        if (face_area[neigh_tri] > 0.)
          queue.push_back(neigh_tri);
      }

      // for (uint edge_idx=0; edge_idx<3; edge_idx++) {
      //   uint idx1 = mesh_->polygons[s].vertices[edge_idx];
      //   uint idx2 = mesh_->polygons[s].vertices[(edge_idx+1)%3];

      //   if (idx1 > idx2) {
      //     uint tmp = idx1;
      //     idx1 = idx2;
      //     idx2 = tmp;
      //   }

      //   Edge edge_id(idx1,idx2);
      //   int neigh_tri_idx = -1;
      //   float max_surface_area = -1.;
      //   for (uint i=0; i<edge_to_triangle[edge_id].size(); i++) {

      //     if (node_face_association_[edge_to_triangle[edge_id][i]] != -1) {
      //       uint node_idx2 = node_face_association_[edge_to_triangle[edge_id][i]];
      //       adj_mat[node_idx*nodes_nb_ + node_idx2] = true;
      //       adj_mat[node_idx2*nodes_nb_ + node_idx] = true;
      //     }

      //     if (visited[edge_to_triangle[edge_id][i]]) {
      //       continue;
      //     } else {
      //       visited[edge_to_triangle[edge_id][i]] = true;
      //     }

      //     // This is not the triangle you're looking for
      //     if (edge_to_triangle[edge_id][i] == s)
      //       continue;

      //     // If we're here, it is one of the unvisited neighboring triangles
      //     if (face_area[edge_to_triangle[edge_id][i]] > max_surface_area) {
      //       neigh_tri_idx = edge_to_triangle[edge_id][i];
      //       max_surface_area = face_area[edge_to_triangle[edge_id][i]];
      //     }
      //   } // --loop over edge neighbors

      //   // If a unvisited triangle was found for that edge, enqueue it
      //   if (neigh_tri_idx != -1 && max_surface_area > 0.)
      //     queue.push_back(neigh_tri_idx);

      // } // --loop over triangle edges


    } // while queue not empty


    // If the node sampled is too small, undo the sampling and the adjacency and try again
    if (nodes_elts_[node_idx].size() < min_node_size) {
      nodes_elts_[node_idx].clear();
      nodes_elts_[node_idx].reserve(min_node_size);
      sampled_indices_.pop_back();

      for (uint i=0; i<nodes_nb_; i++) {
        adj_mat[node_idx*nodes_nb_ + i] = false;
        adj_mat[i*nodes_nb_ + node_idx] = false;
      }

      if (debug_)
        std::cout << "remaining total area " << total_area << std::endl;

      node_idx--;
      continue;
    }

    if (debug_)
      std::cout << node_idx << " Sampled idx " << sampled_idx << " (Contains " << nodes_elts_[node_idx].size() << " faces)" << std::endl;
  } // for each node

  // Fill in the adjacency map
  for (uint i=0; i<node_face_association_.size(); i++) {
    for (uint ni=0; ni<node_face_association_[i].size(); ni++) {
      int node_idx1 = node_face_association_[i][ni];
      for (uint nj=ni+1; nj<node_face_association_[i].size(); nj++) {
        int node_idx2 = node_face_association_[i][nj];
        adj_mat[node_idx1*nodes_nb_ + node_idx2] = true;
        adj_mat[node_idx2*nodes_nb_ + node_idx1] = true;
      }
    }
  }


  // --- Get the vertices associated with each node ---------------------------
  nodes_vertices_.resize(nodes_nb_);
  for (uint node_idx=0; node_idx<nodes_elts_.size(); node_idx++) {
    std::set<uint> vertex_subset;
    for (uint tri_idx=0; tri_idx < nodes_elts_[node_idx].size(); tri_idx++) {
      uint face_idx = nodes_elts_[node_idx][tri_idx];

      vertex_subset.insert(mesh_->polygons[face_idx].vertices[0]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[1]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[2]);
    }

    nodes_vertices_[node_idx].resize(vertex_subset.size());

    uint new_idx = 0;
    for (auto vertex_idx : vertex_subset) {
      nodes_vertices_[node_idx][new_idx] = vertex_idx;
      new_idx++;
    }

  }


  // TODO =====================================================================
  // Fill in the vertex to node association
  node_vertex_association_.resize(pc_->points.size());
  for (uint i=0; i<node_vertex_association_.size(); i++)
    node_vertex_association_[i].resize(nodes_nb_, false);

  for (uint i=0; i<node_face_association_.size(); i++) {
    for (uint j=0; j<3; j++) {
      uint vertex_idx = mesh_->polygons[i].vertices[j];
      for (uint k=0; k<node_face_association_[i].size(); k++) {
        uint node_idx = node_face_association_[i][k];
        node_vertex_association_[vertex_idx][node_idx] = true;
      }
    }
  }

  // \TODO =====================================================================

  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++) {
    valid_indices_[i] = true;
    adj_mat[i*nodes_nb_ + i] = true;
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::computePcaLrf() {
  lrf_.resize(nodes_elts_.size());
  nodes_mean_.resize(nodes_elts_.size());

  for (uint node_idx=0; node_idx<nodes_elts_.size(); node_idx++) {
    if (nodes_vertices_[node_idx].size() < 3)
      continue;

    pcl::IndicesPtr indices(new std::vector<int>(nodes_vertices_[node_idx]));

    pcl::PCA<PointT> pca;
    pca.setInputCloud(pc_);
    pca.setIndices(indices);

    // Eigen::Matrix3f eigvecs = pca.getEigenVectors();
    lrf_[node_idx] = pca.getEigenVectors().transpose();
    lrf_[node_idx].row(1).matrix () = lrf_[node_idx].row(2).cross (lrf_[node_idx].row(0));

    // Eigen::Vector4f mean = pca.getMean();
    nodes_mean_[node_idx] = pca.getMean().head<3>();
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

    if (std::isnan(v1(0)) || std::isnan(v1(1)) || std::isnan(v1(2)))
      continue;

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector3f v21 = v2 - v1;
      // v21.normalize();

      if (std::isnan(v21(0)) || std::isnan(v21(1)) || std::isnan(v21(2)))
        continue;

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
void GraphConstructor::tconvEdgeFeatures(int* tconv_idx) {
  ScopeTime t("TConv edge features computation", debug_);


  if (lrf_.size() == 0)
    computePcaLrf();

  for (uint node_idx=0; node_idx < nodes_elts_.size(); node_idx++) {

    // Create the faces matrix
    std::unordered_map<uint, uint> reverse_vertex_idx;
    for (uint vert_idx=0; vert_idx < nodes_vertices_[node_idx].size(); vert_idx++) {
      reverse_vertex_idx[nodes_vertices_[node_idx][vert_idx]] = vert_idx;
    }

    Eigen::MatrixXi F;
    F.resize(nodes_elts_[node_idx].size(), 3);
    for (uint tri_idx=0; tri_idx<nodes_elts_[node_idx].size(); tri_idx++) {
      uint triangle = nodes_elts_[node_idx][tri_idx];
      F(tri_idx, 0) = reverse_vertex_idx[mesh_->polygons[triangle].vertices[0]];
      F(tri_idx, 1) = reverse_vertex_idx[mesh_->polygons[triangle].vertices[1]];
      F(tri_idx, 2) = reverse_vertex_idx[mesh_->polygons[triangle].vertices[2]];
    }


    // Get the boundary loop
    Eigen::VectorXi bnd;
    igl::boundary_loop(F,bnd);

    if (bnd.size() == 0) {
      if (debug_)
        std::cout << "bnd.size() " << bnd.size() << std::endl;

      valid_indices_[node_idx] = false;
      continue;
    }


    std::vector<std::vector<int> > node_boundary_votes(8, std::vector<int>(nodes_nb_, 0));
    uint boundary_split_size = static_cast<uint>(bnd.size() / 8);


    // Find index of maximum value in the new LRF
    int idx_max_val=-1;
    double max_val=-1e3;
    for (uint i=0; i<bnd.size(); i++) {
      int idx = bnd(i);
      int vert_idx = nodes_vertices_[node_idx][idx];
      Eigen::Vector3f vertex = pc_->points[vert_idx].getVector3fMap();

      if (vertex.dot(lrf_[node_idx].row(0)) > max_val) {
        max_val = vertex.dot(lrf_[node_idx].row(0));
        idx_max_val = idx;
      }
    }


    // Fill in the TConv grid
    for (uint grid_idx=0; grid_idx<8; grid_idx++) {
      for (uint cell_idx=0; cell_idx<boundary_split_size; cell_idx++) {
        uint loop_idx = (idx_max_val + cell_idx + boundary_split_size*grid_idx) % bnd.size();
        uint cur_vertex = nodes_vertices_[node_idx][bnd(loop_idx)];

        for (uint neigh_node_idx=0; neigh_node_idx<nodes_nb_; neigh_node_idx++) {
          if (neigh_node_idx == node_idx)
            continue;

          if (node_vertex_association_[cur_vertex][neigh_node_idx])
            node_boundary_votes[grid_idx][neigh_node_idx]++;
        }
      }
    }

    tconv_idx[node_idx*9 + 4] = node_idx;

    for (uint i=0; i<8; i++) {
      int neigh_idx = node_idx;
      int max_votes = 0;
      for (uint node_idx=0; node_idx<nodes_nb_; node_idx++) {
        if (node_boundary_votes[i][node_idx] > max_votes) {
          max_votes = node_boundary_votes[i][node_idx];
          neigh_idx = node_idx;
        }
      }
      if (i < 4)
        tconv_idx[node_idx*9 + i] = neigh_idx;
      else
        tconv_idx[node_idx*9 + i + 1] = neigh_idx;
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
void GraphConstructor::sphNodeFeatures(double** result, int* tconv_idx, uint image_size, uint num_channels, SphParams sph_params) {
  ScopeTime t("SPH features computation", debug_);

  // Compute the TConv indices
  if (sph_params.tconv_idx)
    tconvEdgeFeatures(tconv_idx);

  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {

    // --- SUBSET EXTRACTION --------------------------------------------------
    std::vector<uint> vertex_idx_mapping;

    std::set<uint> vertex_subset;
    for (uint tri_idx=0; tri_idx < nodes_elts_[node_idx].size(); tri_idx++) {
      uint face_idx = nodes_elts_[node_idx][tri_idx];

      vertex_subset.insert(mesh_->polygons[face_idx].vertices[0]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[1]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[2]);
    }

    // Re-map vertices of the subset to 0-vertices_nb to re-index the triangles properly
    std::unordered_map<uint, uint> reverse_vertex_idx;
    vertex_idx_mapping.resize(vertex_subset.size());
    uint new_idx = 0;
    for (auto vertex_idx : vertex_subset) {
      reverse_vertex_idx[vertex_idx] = new_idx;
      vertex_idx_mapping[new_idx] = vertex_idx;
      new_idx++;
    }


    // --- V F setup ----------------------------------------------------------
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    V.resize(vertex_subset.size(), 3);
    F.resize(nodes_elts_[node_idx].size(), 3);

    // Fill in V
    uint v_idx=0;
    for (auto vertex_idx : vertex_subset) {
      V(v_idx,0) = pc_->points[vertex_idx].x;
      V(v_idx,1) = pc_->points[vertex_idx].y;
      V(v_idx,2) = pc_->points[vertex_idx].z;
      v_idx++;
    }

    // Fill in F
    for (uint loop_idx=0; loop_idx < nodes_elts_[node_idx].size(); loop_idx++) {
      uint face_idx = nodes_elts_[node_idx][loop_idx];
      F(loop_idx, 0) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[0]];
      F(loop_idx, 1) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[1]];
      F(loop_idx, 2) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[2]];
    }


    // --- LRF COMPUTATION ----------------------------------------------------
    Eigen::MatrixXd V_centered;
    Eigen::Matrix3d rf;
    V_centered = V.rowwise() - V.colwise().mean();
    Eigen::MatrixXd cov = (V_centered.adjoint() * V_centered) / double(V.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (cov);

    rf.row (0).matrix () = solver.eigenvectors().col (2);
    rf.row (2).matrix () = solver.eigenvectors().col (0);
    rf.row (1).matrix () = rf.row (2).cross (rf.row (0));

    if (debug_)
      igl::writePLY("./extracted_subset.ply", V, F);

    // Fix two points on the boundary
    Eigen::VectorXi bnd,b(2,1);
    igl::boundary_loop(F,bnd);

    if (bnd.size() == 0) {
      if (debug_)
        std::cout << "bnd.size() " << bnd.size() << std::endl;

      valid_indices_[node_idx] = false;
      continue;
    }

    int idx_min_val=-1, idx_max_val=-1;
    double min_val=1e3, max_val=-1e3;
    for (uint i=0; i<bnd.size(); i++) {
      int idx = bnd(i);
      if (V_centered.row(idx).dot(rf.row(0)) > max_val) {
        max_val = V_centered.row(idx).dot(rf.row(0));
        idx_max_val = idx;
      }

      if (V_centered.row(idx).dot(rf.row(0)) < min_val) {
        min_val = V_centered.row(idx).dot(rf.row(0));
        idx_min_val = idx;
      }
    }


    // --- LSCM COMPUTATION ---------------------------------------------------
    Eigen::MatrixXd V_uv;
    if (sph_params.lscm) {
      b(0) = idx_min_val;
      b(1) = idx_max_val;
      Eigen::MatrixXd bc(2,2);
      bc<<0,0,0,1;

      // LSCM parametrization
      igl::lscm(V,F,b,bc,V_uv);

      // Scale the uv
      V_uv *= 5;

      if (std::isnan(V_uv(0,0)) || std::isnan(V_uv(1,0)) || std::isnan(V_uv(2,0)) ) {
        if (debug_)
          std::cout << "Something's rotten in V_uville" << std::endl;
        valid_indices_[node_idx] = false;
        continue;
      }
    }


    // --- RASTERIZATION ------------------------------------------------------
    // Compute the distance to the tangential plane
    Eigen::VectorXd Vpd = V_centered * rf.row(2).transpose();

    Eigen::MatrixXd W0 = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);
    Eigen::MatrixXd W2 = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);
    Eigen::MatrixXi I_face_idx = Eigen::MatrixXi::Constant(image_size+1, image_size+1, -1);
    Eigen::MatrixXd image_mask = Eigen::MatrixXd::Constant(image_size+1, image_size+1, 0.);
    Eigen::VectorXd V_z;

    if (sph_params.lscm) {
      V_z = Eigen::VectorXd::Constant(V_uv.rows(), 0.);
    } else {
      V_uv = V_centered * rf.block(0, 0, 2, 3).transpose();
      V_z =  -Vpd;
    }

    double min_u = V_uv.col(0).minCoeff(), max_u = V_uv.col(0).maxCoeff();
    double min_v = V_uv.col(1).minCoeff(), max_v = V_uv.col(1).maxCoeff();
    double min_px=std::min(min_u, min_v), max_px=std::max(max_u, max_v);

    rasterize(F, V_uv, V_z, W0, W1, W2, I_face_idx, image_mask, image_size, min_px, max_px);


    // --- FEATURE IMAGE COMPUTATION ----------------------------------------
    Eigen::VectorXd Ved, Vxc, Vyc;

    // Compute the euclidean distance
    if (sph_params.euclidean_distance)
      Ved = V_centered.rowwise().lpNorm<2>();

    if (sph_params.x_coords)
      Vxc = V_centered * rf.row(0).transpose();

    if (sph_params.y_coords)
      Vyc = V_centered * rf.row(1).transpose();

    for (uint i=0; i<image_size; i++) {
      for (uint j=0; j<image_size; j++) {
        if (!image_mask(i, j))
          continue;

        uint face_idx = I_face_idx(i, j);
        uint cur_channel = 0;

        if (sph_params.mask) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = image_mask(i, j);
          cur_channel++;
        }
        if (sph_params.plane_distance) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*Vpd(F(face_idx, 0))
                                                                                     + W1(i, j)*Vpd(F(face_idx, 1))
                                                                                     + W2(i, j)*Vpd(F(face_idx, 2));
          cur_channel++;
        }
        if (sph_params.euclidean_distance) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*Ved(F(face_idx, 0), 2)
                                                                                     + W1(i, j)*Ved(F(face_idx, 1), 2)
                                                                                     + W2(i, j)*Ved(F(face_idx, 2), 2);
          cur_channel++;
        }
        if (sph_params.z_height) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*V(F(face_idx, 0), 2)
                                                                                     + W1(i, j)*V(F(face_idx, 1), 2)
                                                                                     + W2(i, j)*V(F(face_idx, 2), 2);
          cur_channel++;
        }
        if (sph_params.z_rel) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*V_centered(F(face_idx, 0), 2)
                                                                                     + W1(i, j)*V_centered(F(face_idx, 1), 2)
                                                                                     + W2(i, j)*V_centered(F(face_idx, 2), 2);
          cur_channel++;
        }
        if (sph_params.x_coords) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*Vxc(F(face_idx, 0))
                                                                                     + W1(i, j)*Vxc(F(face_idx, 1))
                                                                                     + W2(i, j)*Vxc(F(face_idx, 2));
          cur_channel++;
        }
        if (sph_params.y_coords) {
          result[node_idx][i*image_size*num_channels + j*num_channels + cur_channel] = W0(i, j)*Vyc(F(face_idx, 0))
                                                                                     + W1(i, j)*Vyc(F(face_idx, 1))
                                                                                     + W2(i, j)*Vyc(F(face_idx, 2));
          cur_channel++;
        }
      }
    } // Writing the features into the result image

  } // for loop over each node
} // GraphConstructor::sphNodeFeatures


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::pointProjNodeFeatures(double** result, int* tconv_idx, uint image_size) {
  ScopeTime t("Point Projection features computation", debug_);

  // Compute the TConv indices
  tconvEdgeFeatures(tconv_idx);

  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {
    Eigen::MatrixXd V;
    Eigen::MatrixXd V_centered;
    Eigen::MatrixXi F;
    std::vector<uint> vertex_idx_mapping;

    // --- SUBSET EXTRACTION --------------------------------------------------
    // Extract the proper vertex subset corresponding to our face subset
    std::set<uint> vertex_subset;
    for (uint tri_idx=0; tri_idx < nodes_elts_[node_idx].size(); tri_idx++) {
      uint face_idx = nodes_elts_[node_idx][tri_idx];

      vertex_subset.insert(mesh_->polygons[face_idx].vertices[0]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[1]);
      vertex_subset.insert(mesh_->polygons[face_idx].vertices[2]);
    }

    // Re-map vertices of the subset to 0-vertices_nb to re-index the triangles properly
    std::unordered_map<uint, uint> reverse_vertex_idx;
    vertex_idx_mapping.resize(vertex_subset.size());
    uint new_idx = 0;
    for (auto vertex_idx : vertex_subset) {
      reverse_vertex_idx[vertex_idx] = new_idx;
      vertex_idx_mapping[new_idx] = vertex_idx;
      new_idx++;
    }

    V.resize(vertex_subset.size(), 3);
    F.resize(nodes_elts_[node_idx].size(), 3);

    if (debug_)
      std::cout << "Node " << node_idx << " / " << sampled_indices_.size()
                << " | Node size (faces): " << nodes_elts_[node_idx].size() << " | "
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
    for (uint loop_idx=0; loop_idx < nodes_elts_[node_idx].size(); loop_idx++) {
      uint face_idx = nodes_elts_[node_idx][loop_idx];
      F(loop_idx, 0) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[0]];
      F(loop_idx, 1) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[1]];
      F(loop_idx, 2) = reverse_vertex_idx[mesh_->polygons[face_idx].vertices[2]];
    }


    // --- LRF COMPUTATION ----------------------------------------------------
    Eigen::Matrix3d rf;
    V_centered = V.rowwise() - V.colwise().mean();
    Eigen::MatrixXd cov = (V_centered.adjoint() * V_centered) / double(V.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver (cov);

    rf.row (0).matrix () = solver.eigenvectors().col (2);
    rf.row (2).matrix () = solver.eigenvectors().col (0);
    rf.row (1).matrix () = rf.row (2).cross (rf.row (0));


    // --- POINT TO PROJECT ---------------------------------------------------
    // std::vector<int> k_indices;
    // std::vector<float> k_sqr_distances;
    // double eucl_neigh_size = 2*V_centered.rowwise().lpNorm<2>().maxCoeff();
    // if (std::isnan(pc_->points[sampled_indices_[node_idx]].x))
    //   continue;

    // std::cout << "eucl_neigh_size: " << eucl_neigh_size << std::endl;

    // tree_->radiusSearch(pc_->points[sampled_indices_[node_idx]],
    //                     eucl_neigh_size, k_indices, k_sqr_distances);

    // std::cout << "k_indices.size()" << k_indices.size() << std::endl;


    // --- PROJECTION ---------------------------------------------------------
    Eigen::VectorXd V_u = V_centered * rf.row(0).transpose();
    Eigen::VectorXd V_v = V_centered * rf.row(1).transpose();
    double min_u = V_u.minCoeff(), max_u = V_u.maxCoeff();
    double min_v = V_v.minCoeff(), max_v = V_v.maxCoeff();
    double min_px = std::min(min_u, min_v), max_px = std::max(max_u, max_v);

    uint num_channels=2;


    Eigen::MatrixXd Z_buffer = Eigen::MatrixXd::Constant(image_size+1, image_size+1, -1.e3);
    // for (uint pt_idx=0; pt_idx<k_indices.size(); pt_idx++) {
    //   PointT pt = pc_->points[k_indices[pt_idx]];
    for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
      PointT pt = pc_->points[pt_idx];
      if (std::isnan(pt.x))
        continue;

      Eigen::Vector3d coords = rf * pt.getVector3fMap().cast<double>();
      coords(2) = -coords(2);

      // if (coords(0) < min_u || coords(0) > max_u ||
      //     coords(1) < min_v || coords(1) > max_v)
      //   continue;

      // uint u_px = image_size * (coords(0) - min_u) / (max_u - min_u);
      // uint v_px = image_size * (coords(1) - min_v) / (max_v - min_v);

      if (coords(0) < min_px || coords(0) > max_px ||
          coords(1) < min_px || coords(1) > max_px)
        continue;

      uint u_px = image_size * (coords(0) - min_px) / (max_px - min_px);
      uint v_px = image_size * (coords(1) - min_px) / (max_px - min_px);

      if (coords(2) > Z_buffer(u_px, v_px) && coords(2) < 0.1) {
        result[node_idx][u_px*image_size*num_channels + v_px*num_channels + 0] = 1.;
        result[node_idx][u_px*image_size*num_channels + v_px*num_channels + 1] = coords(2);
        Z_buffer(u_px, v_px) = coords(2);
      }
    }

  } // -- for each node
} // -- GraphConstructor::projNodeFeatures



void GraphConstructor::coordsSetNodeFeatures(double** result, int* tconv_idx, uint feat_nb, uint num_channels) {
  ScopeTime t("Coords Set features computation", debug_);


  // If necessary, compute the LRF
  if (lrf_.size() == 0)
    computePcaLrf();


  // Compute the TConv indices
  tconvEdgeFeatures(tconv_idx);

  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {

    uint feat_sampled_nb = 0;

    uint vertex_nb = nodes_vertices_[node_idx].size();

    if (debug_ && feat_sampled_nb > vertex_nb)
      std::cout << "Less points in the node than we expect to sample ..." << std::endl;

    for (uint index=0; index < vertex_nb; index++) {
      float rand_idx = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/vertex_nb)) + 1;
      if (rand_idx < feat_nb)
        continue;

      // Fill in the matrix
      Eigen::Vector3f coords = pc_->points[nodes_vertices_[node_idx][index]].getVector3fMap();
      Eigen::Vector3f proj_coords = lrf_[node_idx] * (coords - nodes_mean_[node_idx]);

      result[node_idx][num_channels*feat_sampled_nb + 0] = proj_coords(0);
      result[node_idx][num_channels*feat_sampled_nb + 1] = proj_coords(1);
      result[node_idx][num_channels*feat_sampled_nb + 2] = proj_coords(2);

      if (num_channels == 4) {
        result[node_idx][num_channels*feat_sampled_nb + 3] = coords(2);
      }

      feat_sampled_nb++;

      if (feat_sampled_nb == feat_nb)
        break;
    }


  } // -- for each node
} // -- GraphConstructor::coordsNodeFeatures



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::vizMesh(double* adj_mat, bool viz_small_spheres) {
  // --- Viz ------------------------------------------------------------------

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  {
    ScopeTime t("Mesh viz computation", debug_);
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1., "coords", 0);

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


    std::vector<uint> r, g, b, x_noise, y_noise, z_noise;
    r.resize(sampled_indices_.size());
    g.resize(sampled_indices_.size());
    b.resize(sampled_indices_.size());
    x_noise.resize(sampled_indices_.size());
    y_noise.resize(sampled_indices_.size());
    z_noise.resize(sampled_indices_.size());

    float max_noise = 0.05;
    for (uint i =0; i < sampled_indices_.size(); i++){
      r[i] = rand() % 255;
      b[i] = rand() % 255;
      g[i] = rand() % 255;
      x_noise[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_noise));
      y_noise[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_noise));
      z_noise[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_noise));

    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    //   pcl::PointXYZRGB p;
    //   p.x = pc_->points[pt_idx].x;
    //   p.y = pc_->points[pt_idx].y;
    //   p.z = pc_->points[pt_idx].z;

    //   p.r = 230;
    //   p.g = 230;
    //   p.b = 230;

    //   viz_cloud->points.push_back(p);
    // }

    // for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {
    for (uint node_idx=0; node_idx < 1; node_idx++) {
      Eigen::Vector4f p1 = pc_->points[mesh_->polygons[sampled_indices_[node_idx]].vertices[0]].getVector4fMap ();
      Eigen::Vector4f p2 = pc_->points[mesh_->polygons[sampled_indices_[node_idx]].vertices[1]].getVector4fMap ();
      Eigen::Vector4f p3 = pc_->points[mesh_->polygons[sampled_indices_[node_idx]].vertices[2]].getVector4fMap ();

      p1 += p2;
      p1 += p3;
      p1 /= 3;
      p1.normalize();

      for (uint elt_idx=0; elt_idx<nodes_elts_[node_idx].size(); elt_idx++) {
        uint tri_idx = nodes_elts_[node_idx][elt_idx];
        for (uint i=0; i<3; i++) {
          PointT p = pc_->points[mesh_->polygons[tri_idx].vertices[i]];
          pcl::PointXYZRGB p2;
          p2.r = r[node_idx];
          p2.g = g[node_idx];
          p2.b = b[node_idx];
          p2.x = p.x; // + p1(0);
          p2.y = p.y; // + p1(1);
          p2.z = p.z; // + p1(2);
          viz_cloud->points.push_back(p2);
        }
      }
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(viz_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (viz_cloud, rgb, "cloud");


    // Viz resized mesh
    // pcl::PolygonMesh::Ptr mesh_2(new pcl::PolygonMesh);
    // pcl::PCLPointCloud2 point_cloud2;
    // pcl::toPCLPointCloud2(*pc_, point_cloud2);

    // mesh_2->cloud = point_cloud2;
    // mesh_2->polygons = mesh_->polygons;
    // viewer->addPolygonMesh (*mesh_2);

  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }

}

