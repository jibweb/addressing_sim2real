#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>


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
        int idx2 = sampled_indices[i2];
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
             std::vector<int> & sampled_indices) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  uint to_reserve = 0;
  for (uint i=0; i<nodes_vertices.size(); i++) {
    to_reserve += nodes_vertices[i].size();
  }
  viz_cloud->points.reserve(to_reserve);

  for (uint node_idx=0; node_idx<sampled_indices.size(); node_idx++) {
    uint r = rand() % 255;
    uint g = rand() % 255;
    uint b = rand() % 255;

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
      p.x = pc.points[nodes_vertices[node_idx][pt_idx]].x + 0.4*node_center_pt.x;
      p.y = pc.points[nodes_vertices[node_idx][pt_idx]].y + 0.4*node_center_pt.y;
      p.z = pc.points[nodes_vertices[node_idx][pt_idx]].z + 0.4*node_center_pt.z;
      p.r = r;
      p.g = g;
      p.b = b;

      viz_cloud->points.push_back(p);
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(viz_cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (viz_cloud, rgb, "cloud");
}
