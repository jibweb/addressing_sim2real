#pragma once


// #define GRIDSIZE 64
// #define GRIDSIZE_H GRIDSIZE/2


struct Parameters {
  // Graph structure
  // unsigned int nodes_nb; // KEPT
  // unsigned int feat_nb; // MOVED TO FEAT COMPUTE CALL
  // unsigned int edge_feat_nb; // MOVED TO EDGE_FEAT COMPUTE CALL
  // float min_angle_z_normal; // KEPT
  // float neigh_size; // MOVED TO FEAT COMPUTE CALL  /!!!\ need to be resized by gridsize
  // int neigh_nb; // /!\ Correspond to the number of neighbor of a node when using meshes and the number of points in the neighborhood when using a mesh

  // SWITCH TYPE PARAMS MOVED TO preprocessing.py
  // bool feats_3d;
  // bool edge_feats;
  // bool mesh;
  bool scale;
  // General  DEBUG and GRIDSIZE ONLY
  // unsigned int gridsize;
  // bool viz;
  // bool viz_small_spheres;
  // bool debug;
  // PC transformations   TO REMOVE !!
  float to_remove;
  unsigned int to_keep;
  float occl_pct;
  float noise_std;
  unsigned int rotation_deg;
};


struct SphParams {
  bool mask=false;
  bool plane_distance=false;
  bool euclidean_distance=false;
  bool z_height=false;
  bool z_rel=false;
  bool x_coords=false;
  bool y_coords=false;
  bool tconv_idx=false;
  bool lscm=false;
};


struct VizParams {
  bool curvature=false;
  bool graph_skeleton=false;
  bool lrf=false;
  bool mesh=false;
  bool nodes=false;
  bool normals=false;
};
