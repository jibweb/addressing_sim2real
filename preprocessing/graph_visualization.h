#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>


void getJetColour(float v,
                  const float vmin,
                  const float vmax,
                  pcl::PointXYZRGB & p)
{
   p.r = p.g = p.b = 255;
   float dv;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      p.r = 0;
      p.g = static_cast<int>(255.*(4 * (v - vmin) / dv));
   } else if (v < (vmin + 0.5 * dv)) {
      p.r = 0;
      p.b = static_cast<int>(255.*(1 + 4 * (vmin + 0.25 * dv - v) / dv));
   } else if (v < (vmin + 0.75 * dv)) {
      p.r = static_cast<int>(255.*(4 * (v - vmin - 0.5 * dv) / dv));
      p.b = 0;
   } else {
      p.g = static_cast<int>(255.*(1 + 4 * (vmin + 0.75 * dv - v) / dv));
      p.b = 0;
   }
}


template <class T>
void vizGraphSkeleton(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                      pcl::PointCloud<T> & pc,
                      pcl::PolygonMesh & mesh,
                      double* adj_mat,
                      std::vector<int> & sampled_indices,
                      uint nodes_nb) {
  for (uint i=0; i<sampled_indices.size(); i++) {
    uint tri_idx = sampled_indices[i];
    Eigen::Vector3f v = pc.points[mesh.polygons[tri_idx].vertices[0]].getVector3fMap ();
    v += pc.points[mesh.polygons[tri_idx].vertices[1]].getVector3fMap ();
    v += pc.points[mesh.polygons[tri_idx].vertices[2]].getVector3fMap ();
    v /= 3;

    T p;
    p.x = v(0);
    p.y = v(1);
    p.z = v(2);

    viewer->addSphere<T>(p, 0.01, 1., 0., 0., "sphere_" +std::to_string(tri_idx));

    for (uint i2=i; i2<nodes_nb; i2++) {
      if (adj_mat[nodes_nb*i + i2] > 0.) {
        uint idx2 = sampled_indices[i2];
        if (tri_idx != idx2) {
          Eigen::Vector3f v2 = pc.points[mesh.polygons[idx2].vertices[0]].getVector3fMap ();
          v2 += pc.points[mesh.polygons[idx2].vertices[1]].getVector3fMap ();
          v2 += pc.points[mesh.polygons[idx2].vertices[2]].getVector3fMap ();
          v2 /= 3;

          T p2;
          p2.x = v2(0);
          p2.y = v2(1);
          p2.z = v2(2);
          viewer->addLine<T>(p, p2, 0., 0., 1., "line_" +std::to_string(tri_idx)+std::to_string(idx2));
        }
      }
    }
  }
}


template<class T>
void vizMesh(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
             pcl::PointCloud<T> & pc,
             pcl::PolygonMesh & mesh) {
  pcl::PolygonMesh::Ptr mesh_2(new pcl::PolygonMesh);
  pcl::PCLPointCloud2 point_cloud2;
  pcl::toPCLPointCloud2(pc, point_cloud2);
  mesh_2->cloud = point_cloud2;
  mesh_2->polygons = mesh.polygons;
  viewer->addPolygonMesh (*mesh_2);
}



template<class T>
void vizNodes(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
             pcl::PointCloud<T> & pc,
             pcl::PolygonMesh & mesh,
             std::vector<std::vector<int> > & nodes_vertices,
             std::vector<int> & sampled_indices,
             std::vector<std::vector<uint> > & boundary_loops) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  uint to_reserve = 0;
  for (uint i=0; i<nodes_vertices.size(); i++) {
    to_reserve += nodes_vertices[i].size();
  }
  viz_cloud->points.reserve(to_reserve);

  for (uint node_idx=0; node_idx<sampled_indices.size(); node_idx++) {
    uint r = rand() % 210;
    uint g = rand() % 210;
    uint b = rand() % 210;

    std::vector<bool> is_boundary(pc.points.size(), false);
    for (uint i=0; i<boundary_loops[node_idx].size(); i++)
      is_boundary[boundary_loops[node_idx][i]] = true;

    uint tri_idx = sampled_indices[node_idx];
    Eigen::Vector3f v = pc.points[mesh.polygons[tri_idx].vertices[0]].getVector3fMap ();
    v += pc.points[mesh.polygons[tri_idx].vertices[1]].getVector3fMap ();
    v += pc.points[mesh.polygons[tri_idx].vertices[2]].getVector3fMap ();
    v /= 3;

    T node_center_pt;
    node_center_pt.x = v(0);
    node_center_pt.y = v(1);
    node_center_pt.z = v(2);

    for (uint pt_idx=0; pt_idx < nodes_vertices[node_idx].size(); pt_idx++) {
      pcl::PointXYZRGB p;
      float exploded_view_scale = 0.01;
      p.x = pc.points[nodes_vertices[node_idx][pt_idx]].x + exploded_view_scale*node_center_pt.x;
      p.y = pc.points[nodes_vertices[node_idx][pt_idx]].y + exploded_view_scale*node_center_pt.y;
      p.z = pc.points[nodes_vertices[node_idx][pt_idx]].z + exploded_view_scale*node_center_pt.z;

      if (is_boundary[nodes_vertices[node_idx][pt_idx]]) {
        p.r = r + 45;
        p.g = g + 45;
        p.b = b + 45;
      } else {
        p.r = r;
        p.g = g;
        p.b = b;
      }

      viz_cloud->points.push_back(p);
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(viz_cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (viz_cloud, rgb, "cloud_nodes");
}


template<class T>
void vizFaceAngle(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
                  pcl::PointCloud<T> & pc,
                  pcl::PolygonMesh & mesh,
                  std::vector<float> & face_angle) {
  float max_angle = 3.15f / 3.f;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  viz_cloud->points.resize(mesh.polygons.size());
  for (uint tri_idx=0; tri_idx<mesh.polygons.size(); tri_idx++) {
    Eigen::Vector3f v = pc.points[mesh.polygons[tri_idx].vertices[0]].getVector3fMap();
    v += pc.points[mesh.polygons[tri_idx].vertices[1]].getVector3fMap();
    v += pc.points[mesh.polygons[tri_idx].vertices[2]].getVector3fMap();
    v/= 3;

    viz_cloud->points[tri_idx].x = v(0);
    viz_cloud->points[tri_idx].y = v(1);
    viz_cloud->points[tri_idx].z = v(2);

    getJetColour(face_angle[tri_idx], 0., max_angle, viz_cloud->points[tri_idx]);
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(viz_cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (viz_cloud, rgb, "cloud_curv");
}


template<class T>
void vizLRF(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
            pcl::PointCloud<T> & pc,
            pcl::PolygonMesh & mesh,
            std::vector<int> & sampled_indices,
            std::vector<Eigen::Matrix3f> & lrf) {
  for (uint node_idx=0; node_idx< sampled_indices.size(); node_idx++) {
    uint node_centroid = sampled_indices[node_idx];
    Eigen::Vector4f v = pc.points[mesh.polygons[node_centroid].vertices[0]].getVector4fMap ();
    v += pc.points[mesh.polygons[node_centroid].vertices[1]].getVector4fMap ();
    v += pc.points[mesh.polygons[node_centroid].vertices[2]].getVector4fMap ();
    v /= 3;

    float scale = 0.05;
    pcl::PointXYZ p, p_u, p_v, p_n;
    p.x = v(0);
    p.y = v(1);
    p.z = v(2);

    p_u.x = v(0) + scale*lrf[node_idx](0, 0);
    p_u.y = v(1) + scale*lrf[node_idx](0, 1);
    p_u.z = v(2) + scale*lrf[node_idx](0, 2);

    p_v.x = v(0) + scale*lrf[node_idx](1, 0);
    p_v.y = v(1) + scale*lrf[node_idx](1, 1);
    p_v.z = v(2) + scale*lrf[node_idx](1, 2);

    p_n.x = v(0) + scale*lrf[node_idx](2, 0);
    p_n.y = v(1) + scale*lrf[node_idx](2, 1);
    p_n.z = v(2) + scale*lrf[node_idx](2, 2);

    viewer->addArrow<pcl::PointXYZ, pcl::PointXYZ> (p_u, p, 1., 0., 0., false, "arrow_u_" + std::to_string(node_idx));
    viewer->addArrow<pcl::PointXYZ, pcl::PointXYZ> (p_v, p, 0., 1., 0., false, "arrow_v_" + std::to_string(node_idx));
    viewer->addArrow<pcl::PointXYZ, pcl::PointXYZ> (p_n, p, 0., 0., 1., false, "arrow_n_" + std::to_string(node_idx));

    // Eigen::Matrix4f Trans;
    // Trans.setIdentity();
    // Trans.block<3,3>(0,0) = lrf[node_idx].transpose();
    // Trans.rightCols<1>() = v;

    // Eigen::Affine3f F;
    // F = Trans;
    // viewer->addCoordinateSystem(0.1, F, "lrf_"+std::to_string(node_idx));
  }
}