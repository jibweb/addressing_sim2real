#include <random>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
// #include <time>

#include <boost/functional/hash.hpp>
#include <pcl/common/pca.h>
#include <pcl/conversions.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>


// #define WITH_LIBIGL 0
#ifdef WITH_LIBIGL
  #include <igl/boundary_loop.h>
  #include <igl/lscm.h>
  #include <igl/writePLY.h>
#endif

#include "tinyply.h"
#include "graph_construction.h"
#include "mesh_utils.cpp"
#include "graph_visualization.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// GENERAL ///////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int GraphConstructor::initializeMesh(bool angle_sampling, double* adj_mat, float sampling_target_val) {

  if (!debug_)
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  ScopeTime t("Graph Initialization", debug_);

  {
    ScopeTime t1("Tinyply file opening", debug_);
    loadPLY(filename_, *pc_, *mesh_);
  }


  if (debug_)
    std::cout << "NOT checking for NaN in normals !" << std::endl;
  // for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
  //   if (std::isnan(pc_->points[pt_idx].normal_x) || std::isnan(pc_->points[pt_idx].normal_y) || std::isnan(pc_->points[pt_idx].normal_z))
  //     return -1;
  // }

  if (debug_) {
    std::cout << "PolygonMesh: " << mesh_->polygons.size() << " triangles" << std::endl;
    std::cout << "PC size: " << pc_->points.size() << std::endl;
  }

  // Data augmentation
  scale_points_unit_sphere<PointT> (*pc_, 1.);
  // scale_points_unit_sphere (*pc_, gridsize_/2, centroid);
  // params_.neigh_size = params_.neigh_size * gridsize_/2;
  // augment_data(pc_, params_);



  // TODO : probably should be a param. Weird condition for meshes with triangles of variable areas
  uint min_node_size = 256;
  nodes_elts_.resize(nodes_nb_);
  for (uint i=0; i < nodes_elts_.size(); i++)
    nodes_elts_[i].reserve(min_node_size);

  // TODO Remove nodes with a wrong angle_z_normal

  // Initialize the valid indices
  valid_indices_.resize(nodes_nb_);

  // --- Edge connectivity ----------------------------------------------------
  edge_to_triangle_.reserve(3*mesh_->polygons.size());
  triangle_neighbors_.resize(mesh_->polygons.size());

  {
    ScopeTime t2("Connectivity computation", debug_);
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

        auto arr = edge_to_triangle_.find(edge_id);

        if (arr != edge_to_triangle_.end()) {
          // Edge exists already
          arr->second[1] = tri_idx;
        } else {
          // Edge doesn't exist yet
          edge_to_triangle_[edge_id][0] = tri_idx;
          edge_to_triangle_[edge_id][1] = -1;
        }
      }
    }

    // std::vector<std::vector<uint> > triangle_neighbors_(mesh_->polygons.size());
    for (uint i=0; i<triangle_neighbors_.size(); i++)
      triangle_neighbors_[i].reserve(3);

    for(auto& it : edge_to_triangle_) {
      // TODO slightly shitty way of resolving non manifold meshes
      // if (it.second.size() >= 2) {
      if (it.second[1] != -1) {
        triangle_neighbors_[it.second[0]].push_back(it.second[1]);
        triangle_neighbors_[it.second[1]].push_back(it.second[0]);
      }
    }
  }

  // BFS sampling with area as a stopping criterion
  if (angle_sampling)
    angleBasedNodeSampling(sampling_target_val);
  else
    areaBasedNodeSampling(sampling_target_val);

  {
    ScopeTime t3("Node association", debug_);
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
  }


  // --- Get the vertices associated with each node ---------------------------
  {
    ScopeTime t3("Node vertices", debug_);
    nodes_vertices_.resize(nodes_nb_);
    for (uint node_idx=0; node_idx<nodes_elts_.size(); node_idx++) {
      std::unordered_set<uint> vertex_subset;
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
  }

  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++) {
    valid_indices_[i] = true;
    adj_mat[i*nodes_nb_ + i] = true;
  }

  return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::areaBasedNodeSampling(float target_area) {
  ScopeTime t("Area-based node sampling", debug_);

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

  // float target_area = neigh_size;

  if (debug_)
    std::cout << "Target area: " << target_area << " / Total area: " << total_area  << std::endl; //<< " / Proportion: " << neigh_size << std::endl; //static_cast<float>(neigh_nb) / mesh_->polygons.size() << std::endl;

  // --- BFS surface sampling (area dependent) --------------------------------
  // --- Global setup for the sampling procedure ------------------------------
  struct timeval time;
  gettimeofday(&time,NULL);
  srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
  // srand (static_cast<unsigned int> (time (NULL)));


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

      if (samplable_face_area[s] > (target_area - sampled_area)) {
        total_area = std::max(0.f, total_area  - target_area + sampled_area);
        samplable_face_area[s] = samplable_face_area[s] - (target_area - sampled_area);
      } else {
        total_area = std::max(0.f, total_area  - samplable_face_area[s]);
        samplable_face_area[s] = 0.f;
      }

      sampled_area += face_area[s];


      // For each edge, find the unvisited neighbor and visit all of them
      for (uint neigh_idx=0; neigh_idx<triangle_neighbors_[s].size(); neigh_idx++) {
        uint neigh_tri = triangle_neighbors_[s][neigh_idx];

        if (visited[neigh_tri])
          continue;

        visited[neigh_tri] = true;

        if (face_area[neigh_tri] > 0.)
          queue.push_back(neigh_tri);
      }


    } // while queue not empty


    // If the node sampled is too small, undo the sampling and the adjacency and try again
    // if (nodes_elts_[node_idx].size() < min_node_size) {
    if (sampled_area < target_area/2) {
      nodes_elts_[node_idx].clear();
      sampled_indices_.pop_back();

      if (debug_)
        std::cout << "remaining total area " << total_area << std::endl;

      node_idx--;
      continue;
    } else {
      for (auto elt : nodes_elts_[node_idx])
        node_face_association_[elt].push_back(node_idx);
    }

    if (debug_)
      std::cout << node_idx << " Sampled idx " << sampled_idx << " (Contains " << nodes_elts_[node_idx].size() << " faces)" << std::endl;
  } // for each node
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::angleBasedNodeSampling(float target_angle) {
  ScopeTime t("Angle-based node sampling", debug_);

  float total_angle = 0.f;
  face_angle_.resize(mesh_->polygons.size(), 0.f);
  face_normals_.resize(mesh_->polygons.size());

  // Get the normals per face
  for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
    face_normals_[tri_idx] = pc_->points[mesh_->polygons[tri_idx].vertices[0]].getNormalVector3fMap();
    face_normals_[tri_idx] += pc_->points[mesh_->polygons[tri_idx].vertices[1]].getNormalVector3fMap();
    face_normals_[tri_idx] += pc_->points[mesh_->polygons[tri_idx].vertices[2]].getNormalVector3fMap();

    face_normals_[tri_idx].normalize();
  }

  // Get the average angle per face
  for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
    float avg_angle = 0.f;
    for (uint neigh_idx=0; neigh_idx<triangle_neighbors_[tri_idx].size(); neigh_idx++) {
      uint neigh_tri = triangle_neighbors_[tri_idx][neigh_idx];
      avg_angle += acos(std::min(1.0f, fabs(face_normals_[tri_idx].dot(face_normals_[neigh_tri]))));

      if (std::isnan(avg_angle))
        std::cout << "NaN angle: "
                  << acos(fabs(face_normals_[tri_idx].dot(face_normals_[neigh_idx]))) << ", "
                  << acos(std::min(1.0f, fabs(face_normals_[tri_idx].dot(face_normals_[neigh_idx])))) << ", "
                  << fabs(face_normals_[tri_idx].dot(face_normals_[neigh_idx])) << ", "
                  << face_normals_[tri_idx].dot(face_normals_[neigh_idx]) << ", \n"
                  << face_normals_[tri_idx] << ", \n"
                  << face_normals_[neigh_idx] << " \n" << std::endl;

    }

    if (triangle_neighbors_[tri_idx].size() > 0)
      face_angle_[tri_idx] = avg_angle / static_cast<float>(triangle_neighbors_[tri_idx].size());

    total_angle += face_angle_[tri_idx];
  }


  // Get sorted indices based on the face angle
  // std::vector<uint> sorted_indices(face_angle_.size());
  // for (uint idx=0; idx<face_angle_.size(); idx++)
  //   sorted_indices[idx] = idx;

  // // sort indexes based on comparing values in face_angle_ (max first)
  // std::sort(sorted_indices.begin(), sorted_indices.end(),
  //      [&face_angle_](size_t i1, size_t i2) {return face_angle_[i1] > face_angle_[i2];});


  // === BFS angle sampling  ==================================================
  // --- Seed for random sampling ---------------------------------------------
  struct timeval time;
  gettimeofday(&time,NULL);
  srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
  std::vector<bool> samplable_face(mesh_->polygons.size(), true);
  int total_samplable_face = mesh_->polygons.size();

  // --- Do a BFS per node in the graph ---------------------------------------
  node_face_association_.resize(mesh_->polygons.size());
  for (uint i=0; i<node_face_association_.size(); i++)
    node_face_association_[i].reserve(4);


  if (debug_)
    std::cout << "Target angle: " << target_angle << " / Total angle: " << total_angle  << "\n"
              << "Node \t| Idx  \t| Size\t| Cum. ang.\t| Rem. faces "
              << "\n----------------------------------------------------" << std::endl;

  for (uint node_idx=0; node_idx < nodes_nb_; node_idx++) {
    // --- Select a node and enqueue it ---------------------------------------
    if (total_samplable_face == 0)
      break;

    uint sampled_idx = mesh_->polygons.size() + 1;

    // Purely random node sampling
    int rdn_weight = std::max(static_cast <int> (rand() % total_samplable_face), 1);
    for (uint tri_idx=0; tri_idx<mesh_->polygons.size(); tri_idx++) {
      if (samplable_face[tri_idx])
        rdn_weight--;

      if (rdn_weight <= 0) {
        sampled_idx = tri_idx;
        break;
      }
    }

    // Max angle based node sampling
    // for (uint i=0; i<sorted_indices.size(); i++) {
    //   if (samplable_face[sorted_indices[i]]) {
    //     sampled_idx = sorted_indices[i];
    //     break;
    //   }
    // }

    // Failed to sample a point
    if (sampled_idx == (mesh_->polygons.size() + 1))
      break;

    sampled_indices_.push_back(sampled_idx);

    // --- Setup for BFS ------------------------------------------------------
    std::deque<int> queue;
    queue.push_back(sampled_idx);

    float sampled_angle = 0.;

    std::vector<bool> visited(mesh_->polygons.size(), false);
    std::vector<bool> vertex_visited(pc_->points.size(), false);
    visited[sampled_idx] = true;
    vertex_visited[mesh_->polygons[sampled_idx].vertices[0]] = true;
    vertex_visited[mesh_->polygons[sampled_idx].vertices[1]] = true;
    vertex_visited[mesh_->polygons[sampled_idx].vertices[2]] = true;

    // --- BFS over the graph to extract the neighborhood ---------------------
    while(!queue.empty() && sampled_angle < target_angle) {
      // Dequeue a face
      int s = queue.front();
      queue.pop_front();

      // Update the curv
      nodes_elts_[node_idx].push_back(s);
      if (samplable_face[s])
        total_samplable_face--;
      samplable_face[s] = false;
      sampled_angle += std::min(face_angle_[s], 100.f);

      // For each edge, find the unvisited neighbor and visit all of them
      for (uint neigh_idx=0; neigh_idx<triangle_neighbors_[s].size(); neigh_idx++) {
        uint neigh_tri = triangle_neighbors_[s][neigh_idx];

        if (visited[neigh_tri])
          continue;

        // Prevent the node from wrapping around something and merging on the other side
        // The detection criterion is as follow:
        //   - if the  3 vertices of the neighboring face are already visited AND
        //   - the neighboring triangle has two unvisited neighbors, it's wrapping around
        if (vertex_visited[mesh_->polygons[neigh_tri].vertices[0]] &&
            vertex_visited[mesh_->polygons[neigh_tri].vertices[1]] &&
            vertex_visited[mesh_->polygons[neigh_tri].vertices[2]] &&
            triangle_neighbors_[neigh_tri].size() == 3) {
          uint neigh_neigh_visited = static_cast<int>(visited[triangle_neighbors_[neigh_tri][0]]) +
                                     static_cast<int>(visited[triangle_neighbors_[neigh_tri][1]]) +
                                     static_cast<int>(visited[triangle_neighbors_[neigh_tri][2]]);
          if (neigh_neigh_visited == 1)
            continue;
        }

        vertex_visited[mesh_->polygons[neigh_tri].vertices[0]] = true;
        vertex_visited[mesh_->polygons[neigh_tri].vertices[1]] = true;
        vertex_visited[mesh_->polygons[neigh_tri].vertices[2]] = true;
        visited[neigh_tri] = true;
        queue.push_back(neigh_tri);
      }
    } // while queue not empty

    // If the node sampled is too small, undo the sampling and the adjacency and try again
    if (sampled_angle < target_angle/10.f) {
      nodes_elts_[node_idx].clear();
      sampled_indices_.pop_back();

      if (debug_)
        std::cout << "Deleted node with angle: " << sampled_angle << "/" << target_angle/10.f << ", rem. sampl. faces " << total_samplable_face << std::endl;

      node_idx--;
      continue;
    } else {
      for (auto elt : nodes_elts_[node_idx])
        node_face_association_[elt].push_back(node_idx);
    }

    if (debug_) {
      std::cout << node_idx
                << "\t| " << sampled_idx
                << "\t| " << nodes_elts_[node_idx].size()
                << "\t| " << sampled_angle
                << "\t| " << total_samplable_face
                << std::endl;
    }
  } // for each node

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::extractNodeBoundaries() {
  ScopeTime t("Node boundaries extraction", debug_);
  boundary_loops_.resize(sampled_indices_.size());

  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {
    // --- Prepare the added vector ---------------------------------------------
    std::vector<bool> added(mesh_->polygons.size(), false);
    for (uint i=0; i<nodes_elts_[node_idx].size(); i++)
      added[nodes_elts_[node_idx][i]] = true;


    // --- Find the border vertices from the border faces -----------------------
    std::set<uint> border_vertex;
    std::unordered_map<uint, std::vector<uint> > border_vertex_neighbor;

    for (uint elt_idx=0; elt_idx < nodes_elts_[node_idx].size(); elt_idx++) {
      uint face_idx = nodes_elts_[node_idx][elt_idx];

      for (uint edge_idx=0; edge_idx<3; edge_idx++) {
        uint idx1 = mesh_->polygons[face_idx].vertices[edge_idx];
        uint idx2 = mesh_->polygons[face_idx].vertices[(edge_idx+1)%3];

        if (idx1 > idx2) {
          uint tmp = idx1;
          idx1 = idx2;
          idx2 = tmp;
        }

        Edge edge_id(idx2,idx1);
        auto arr = edge_to_triangle_.find(edge_id);

        if (arr == edge_to_triangle_.end()) {
          std::cout << "Oh my god oh my god oh my god !! This edge was not added yet !!!" << std::endl;
          continue;
        }

        // An actual border of the original mesh OR One of the two neighboring faces isn't part of the node
        if (arr->second[1] == -1 || (!added[arr->second[0]]) || (!added[arr->second[1]])) {
          border_vertex.insert(idx1);
          border_vertex.insert(idx2);
          border_vertex_neighbor[idx1].push_back(idx2);
          border_vertex_neighbor[idx2].push_back(idx1);
        }
      }
    }

    if (border_vertex.size() == 0)
      return;

    for (auto border_vert : border_vertex)
      boundary_loops_[node_idx].push_back(border_vert);


    // --- Follow the loop from border vertices ---------------------------------
    std::unordered_map<Edge, bool, boost::hash<Edge>> visited;
    std::vector<std::vector<uint> > boundary_loops;
    uint loop_idx = 0;
    uint full_size = 0;
    uint max_loop=0, max_loop_idx=0;

    do {
      boundary_loops.push_back( std::vector<uint>() );

      // Extract the first node with unvisited edges
      uint start_vertex = *border_vertex.begin();

      for (auto border_vert : border_vertex) {
        for (uint neigh_idx=0; neigh_idx < border_vertex_neighbor[border_vert].size(); neigh_idx++) {
          uint neigh = border_vertex_neighbor[border_vert][neigh_idx];

          uint idx1, idx2;
          if (neigh > border_vert) {
            idx1 = border_vert;
            idx2 = neigh;
          } else {
            idx1 = neigh;
            idx2 = border_vert;
          }

          Edge edge_id(idx2,idx1);

          if (!visited[edge_id]) {
            start_vertex = border_vert;
            break;
          }
        }
      }

      // From that first node, follow the boundary
      uint cur_vertex = start_vertex;
      uint prev_vertex = start_vertex;

      do {
        boundary_loops[loop_idx].push_back(cur_vertex);
        prev_vertex = cur_vertex;

        for (uint neigh_idx=0; neigh_idx < border_vertex_neighbor[cur_vertex].size(); neigh_idx++) {
          uint neigh = border_vertex_neighbor[cur_vertex][neigh_idx];

          uint idx1, idx2;
          if (neigh > cur_vertex) {
            idx1 = cur_vertex;
            idx2 = neigh;
          } else {
            idx1 = neigh;
            idx2 = cur_vertex;
          }

          Edge edge_id(idx2,idx1);

          if (!visited[edge_id] && neigh != prev_vertex) {
            cur_vertex = neigh;
            visited[edge_id] = true;
            break;
          }
        }
      } while (cur_vertex != start_vertex && cur_vertex != prev_vertex);

      if (boundary_loops[loop_idx].size() > max_loop) {
        max_loop = boundary_loops[loop_idx].size();
        max_loop_idx = loop_idx;
      }

      full_size += boundary_loops[loop_idx].size();
      loop_idx++;

    } while (full_size < border_vertex.size());

    // std::cout << "Boundary loop vertices: " << boundary_loops[max_loop_idx].size() << " / " << full_size << " / " << border_vertex.size() << std::endl;

    boundary_loops_[node_idx].resize(boundary_loops[max_loop_idx].size());
    for (uint i=0; i<boundary_loops[max_loop_idx].size(); i++)
      boundary_loops_[node_idx][i] = boundary_loops[max_loop_idx][i];
  } // endfor node_idx in sampled_indices
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::computeNormalAlignedPcaLrf() {
  ScopeTime t("PCA LRF computation", debug_);
  lrf_.resize(nodes_elts_.size());
  zlrf_.resize(nodes_elts_.size());
  nodes_mean_.resize(nodes_elts_.size());
  Eigen::Vector3f z(0.f, 0.f, 1.f);

  for (uint node_idx=0; node_idx<nodes_elts_.size(); node_idx++) {
    if (nodes_vertices_[node_idx].size() < 3)
      continue;

    pcl::IndicesPtr indices(new std::vector<int>(nodes_vertices_[node_idx]));

    pcl::PCA<PointT> pca;
    pca.setInputCloud(pc_);
    pca.setIndices(indices);

    lrf_[node_idx] = pca.getEigenVectors().transpose();

    // Align with normals
    uint sampled_nb = 100;
    uint plusNormals = 0;
    for (uint i=0; i<sampled_nb; i++) {
      uint idx = static_cast<uint>(rand() % nodes_vertices_[node_idx].size());
      uint pt_idx = nodes_vertices_[node_idx][idx];
      Eigen::Vector3f v = pc_->points[pt_idx].getVector3fMap();

      if (v.dot(lrf_[node_idx].row(2)) > 0.)
        plusNormals++;
    }

    // If less than half aligns, flip the LRF
    if (2*plusNormals < sampled_nb)
      lrf_[node_idx].row(2) = -lrf_[node_idx].row(2);

    // Update the properties of the graph
    lrf_[node_idx].row(1).matrix () = lrf_[node_idx].row(2).cross (lrf_[node_idx].row(0));
    nodes_mean_[node_idx] = pca.getMean().head<3>();


    // Z-based LRF
    Eigen::Vector3f node_normal = lrf_[node_idx].row(2);
    node_normal(2) = 0.f;
    node_normal.normalize();
    zlrf_[node_idx].row(0) = node_normal;
    zlrf_[node_idx].row(1) = z.cross(node_normal);
    zlrf_[node_idx].row(2) = z;
  }
}



void GraphConstructor::computeVerticesCurvature() {
  ScopeTime t("Vertices curvature computation", debug_);
  // Edge curvature
  std::unordered_map<Edge, float, boost::hash<Edge> > edge_curv;
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

      auto arr = edge_curv.find(edge_id);

      if (arr == edge_curv.end()) {
        Eigen::Vector3f v1 = pc_->points[idx1].getVector3fMap();
        Eigen::Vector3f v2 = pc_->points[idx2].getVector3fMap();
        Eigen::Vector3f n1 = pc_->points[idx1].getNormalVector3fMap();
        Eigen::Vector3f n2 = pc_->points[idx2].getNormalVector3fMap();
        float signed_curv = (n1 - n2).dot(v1 - v2);
        signed_curv /= (v1 - v2).squaredNorm();
        edge_curv[edge_id] = signed_curv;
      }
    }
  }

  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    pc_->points[pt_idx].curvature = 0.f;
  }

  // Accumulate curvature for each vertex based on the edge curvature it belongs to
  std::vector<int> vertex_edge_count(pc_->points.size(), 0);
  for (auto& it : edge_curv){
    pc_->points[it.first.first].curvature += fabs(it.second);
    pc_->points[it.first.second].curvature += fabs(it.second);
    vertex_edge_count[it.first.first]++;
    vertex_edge_count[it.first.second]++;
  }

  // Normalize curv by number of edge involved
  for (uint pt_idx=0; pt_idx<pc_->points.size(); pt_idx++) {
    pc_->points[pt_idx].curvature /= vertex_edge_count[pt_idx];
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

  // TODO: NEEDS TO BE REWRITTEN FOR sampled_indices BEING FACES AND NOT VERTICES
  // int index1, index2;

  // for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
  //   index1 = sampled_indices_[pt1_idx];
  //   Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();

  //   if (std::isnan(v1(0)) || std::isnan(v1(1)) || std::isnan(v1(2)))
  //     continue;

  //   for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
  //     index2 = sampled_indices_[pt2_idx];
  //     Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
  //     Eigen::Vector3f v21 = v2 - v1;
  //     // v21.normalize();

  //     if (std::isnan(v21(0)) || std::isnan(v21(1)) || std::isnan(v21(2)))
  //       continue;

  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 0] = v21(0);
  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 1] = v21(1);
  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 2] = v21(2);
  //   }
  // }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::rotZEdgeFeatures(double* edge_feats, float min_angle_z_normal) {

  // TODO: NEEDS TO BE REWRITTEN FOR sampled_indices BEING FACES AND NOT VERTICES
  // int index1, index2;
  // Eigen::Vector3f z(0., 0., 1.);
  // z.normalize();

  // for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
  //   index1 = sampled_indices_[pt1_idx];
  //   Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();
  //   Eigen::Vector3f n1 = pc_->points[index1].getNormalVector3fMap();
  //   n1.normalize();

  //   if (acos(fabs(n1.dot(z))) < min_angle_z_normal*M_PI/180.) {
  //     valid_indices_[pt1_idx] = false;
  //     continue;
  //   }


  //   Eigen::Vector3f n_axis;
  //   n_axis(0) = n1(0);
  //   n_axis(1) = n1(1);
  //   n_axis(2) = 0.;
  //   n_axis.normalize();
  //   Eigen::Vector3f w = n_axis.cross(z);

  //   Eigen::Matrix3f lrf;

  //   lrf.row(0) << n1;
  //   lrf.row(1) << w;
  //   lrf.row(2) << z;

  //   for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
  //     index2 = sampled_indices_[pt2_idx];
  //     Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
  //     Eigen::Vector3f v21 = v2 - v1;
  //     Eigen::Vector3f local_coords = lrf * v21;
  //     // v21.normalize();

  //     if (std::isnan(local_coords(0)) || std::isnan(local_coords(1)) || std::isnan(local_coords(2))) {
  //       std::cout << local_coords << "\n----\n"
  //                 << lrf << "\n---\n"
  //                 << n1 << "\n---\n"
  //                 << n_axis << "\n---\n"
  //                 << w << "\n---\n"
  //                 << z << "\n***\n"
  //                 << std::endl;
  //     }

  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 0] = local_coords(0);
  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 1] = local_coords(1);
  //     edge_feats[3*(nodes_nb_*pt1_idx + pt2_idx) + 2] = local_coords(2);
  //   }
  // }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::tconvEdgeFeatures(int* tconv_idx) {
  // TODO =====================================================================
  // Fill in the vertex to node association
  node_vertex_association_.resize(pc_->points.size(), std::vector<bool>(nodes_nb_, false));
  // node_vertex_association_.resize(pc_->points.size());
  // for (uint i=0; i<node_vertex_association_.size(); i++)
  //   node_vertex_association_[i].resize(nodes_nb_, false);

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


  ScopeTime t("TConv edge features computation", debug_);

  if (lrf_.size() == 0)
    computeNormalAlignedPcaLrf();

  if (boundary_loops_.size() == 0)
    extractNodeBoundaries();

  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {

    Eigen::VectorXi bnd;
    bnd.resize(boundary_loops_[node_idx].size());
    for (uint i=0; i < boundary_loops_[node_idx].size(); i++)
      bnd(i) = boundary_loops_[node_idx][i];

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
      int vert_idx = bnd(i);
      Eigen::Vector3f vertex = pc_->points[vert_idx].getVector3fMap();

      if (vertex.dot(lrf_[node_idx].row(0)) > max_val) {
        max_val = vertex.dot(lrf_[node_idx].row(0));
        idx_max_val = i;
      }
    }


    // Fill in the TConv grid
    for (uint grid_idx=0; grid_idx<8; grid_idx++) {
      for (uint cell_idx=0; cell_idx<boundary_split_size; cell_idx++) {
        uint loop_idx = (idx_max_val + cell_idx + boundary_split_size*grid_idx) % bnd.size();
        // uint cur_vertex = nodes_vertices_[node_idx][bnd(loop_idx)];
        uint cur_vertex = bnd(loop_idx);

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
#ifdef WITH_LIBIGL
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
#endif
} // GraphConstructor::sphNodeFeatures


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::pointProjNodeFeatures(double** result, int* tconv_idx, uint image_size) {
#ifdef WITH_LIBIGL
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
#endif
} // GraphConstructor::projNodeFeatures



void GraphConstructor::coordsSetNodeFeatures(double** result, uint feat_nb, uint num_channels) {
  ScopeTime t("Coords Set features computation", debug_);


  // If necessary, compute the LRF
  if (lrf_.size() == 0)
    computeNormalAlignedPcaLrf();


  for (uint node_idx=0; node_idx < sampled_indices_.size(); node_idx++) {
    uint face_nb = nodes_elts_[node_idx].size();
    float scale = 0.f;
    for (uint feat_idx=0; feat_idx<feat_nb; feat_idx++) {
      uint rand_idx = rand() % face_nb;

      float u = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
      float v = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
      float w = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
      float uvw = u+v+w;
      u /= uvw;
      v /= uvw;
      w /= uvw;

      Eigen::Vector3f p0 = pc_->points[mesh_->polygons[nodes_elts_[node_idx][rand_idx]].vertices[0]].getVector3fMap();
      Eigen::Vector3f p1 = pc_->points[mesh_->polygons[nodes_elts_[node_idx][rand_idx]].vertices[1]].getVector3fMap();
      Eigen::Vector3f p2 = pc_->points[mesh_->polygons[nodes_elts_[node_idx][rand_idx]].vertices[2]].getVector3fMap();

      Eigen::Vector3f coords = u*p0 + v*p1 + w*p2;
      // Eigen::Vector3f proj_coords = lrf_[node_idx] * (coords - nodes_mean_[node_idx]);
      Eigen::Vector3f proj_coords = zlrf_[node_idx] * (coords - nodes_mean_[node_idx]);

      if (proj_coords.norm() > scale)
        scale = proj_coords.norm();

      result[node_idx][num_channels*feat_idx + 0] = proj_coords(0);
      result[node_idx][num_channels*feat_idx + 1] = proj_coords(1);
      result[node_idx][num_channels*feat_idx + 2] = proj_coords(2);

      if (num_channels == 4)
        result[node_idx][num_channels*feat_idx + 3] = coords(2);
    }

    for (uint feat_idx=0; feat_idx<feat_nb; feat_idx++) {
      result[node_idx][num_channels*feat_idx + 0] /= scale;
      result[node_idx][num_channels*feat_idx + 1] /= scale;
      result[node_idx][num_channels*feat_idx + 2] /= scale;
    }
  } // -- for each node
} // -- GraphConstructor::coordsNodeFeatures



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GraphConstructor::vizGraph(double* adj_mat, VizParams viz_params) {
  // --- Viz ------------------------------------------------------------------

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

  {
    ScopeTime t("Visualization setup", debug_);
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1., "coords", 0);

    if (viz_params.curvature)
      vizFaceAngle(viewer, *pc_, *mesh_, face_angle_);

    if (viz_params.graph_skeleton)
      vizGraphSkeleton<PointT>(viewer, *pc_, *mesh_, adj_mat, sampled_indices_, nodes_nb_);

    if (viz_params.lrf) {
      // If necessary, compute the LRF
      if (lrf_.size() == 0)
        computeNormalAlignedPcaLrf();
      vizLRF(viewer, *pc_, *mesh_, sampled_indices_, lrf_);
    }

    if (viz_params.normals)
      viewer->addPointCloudNormals<PointT, PointT> (pc_, pc_, 1, 0.05, "normals");

    if (viz_params.mesh)
      vizMesh<PointT>(viewer, *pc_, *mesh_);

    if (viz_params.nodes) {
      if (boundary_loops_.size() == 0)
        extractNodeBoundaries();
      vizNodes<PointT>(viewer, *pc_, *mesh_, nodes_vertices_, sampled_indices_, boundary_loops_);
    }
  }

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}

