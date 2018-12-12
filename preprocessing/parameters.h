#pragma once


// #define GRIDSIZE 64
// #define GRIDSIZE_H GRIDSIZE/2


struct Parameters
{
  // Graph structure
  // unsigned int nodes_nb; // KEPT
  // unsigned int feat_nb; // MOVED TO FEAT COMPUTE CALL
  // unsigned int edge_feat_nb; // MOVED TO EDGE_FEAT COMPUTE CALL
  float min_angle_z_normal; // KEPT
  float neigh_size; // MOVED TO FEAT COMPUTE CALL  /!!!\ need to be resized by gridsize
  // int neigh_nb; // /!\ Correspond to the number of neighbor of a node when using meshes and the number of points in the neighborhood when using a mesh

  // SWITCH TYPE PARAMS MOVED TO preprocessing.py
  // bool feats_3d;
  // bool edge_feats;
  bool mesh;
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
