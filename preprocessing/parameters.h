#pragma once


// #define GRIDSIZE 64
// #define GRIDSIZE_H GRIDSIZE/2


struct TransfoParams {
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
