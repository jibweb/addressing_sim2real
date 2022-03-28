#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <pcl/conversions.h>
#include <pcl/PolygonMesh.h>
#include <pcl/Vertices.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include "scope_time.h"
#include "augmentation_preprocessing.cpp"

namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////////////////////////
std::string string_to_hex(const std::string& input) {
    static const char* const lut = "0123456789ABCDEF";
    size_t len = input.length();

    std::string output;
    output.reserve(2 * len);
    for (size_t i = 0; i < len; ++i)
    {
        const unsigned char c = input[i];
        output.push_back(lut[c >> 4]); // right bitwise shift
        output.push_back(lut[c & 15]); // AND bitwise operator
    }
    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double scale_points_unit_sphere (pcl::PointCloud<pcl::PointXYZ> &pc,
                               float scalefactor) {
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid (pc, centroid);
  pcl::demeanPointCloud (pc, centroid, pc);

  float max_distance = 0., d;
  pcl::PointXYZ cog;
  cog.x = 0;
  cog.y = 0;
  cog.z = 0;

  for (size_t i = 0; i < pc.points.size (); ++i)
  {
    d = pcl::euclideanDistance(cog,pc.points[i]);
    if (d > max_distance)
      max_distance = d;
  }

  float scale_factor = 1.0f / max_distance * scalefactor;

  Eigen::Affine3f matrix = Eigen::Affine3f::Identity();
  matrix.scale (scale_factor);
  pcl::transformPointCloud (pc, pc, matrix);

  return static_cast<double>(max_distance);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void readOFF(const std::string filename,
             pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
             std::vector<pcl::Vertices>& triangles) {
  std::string line;
  ifstream off_file (filename);
  if (off_file.is_open()) {
    // Check that it is a proper OFF file
    std::getline (off_file,line);
    if (line.compare("OFF\r") != 0 && line.compare("OFF") != 0) {
      throw std::invalid_argument( "File " + filename + " does not start with OFF ! NOT COOL !" );
    }

    // Extract number of vertices and faces
    std::getline (off_file,line);
    std::vector<std::string> nums;
    boost::split(nums, line, [](char c){return c == ' ';});
    int num_vertices = std::stoi(nums[0]);
    int num_faces = std::stoi(nums[1]);

    pc->points.resize(num_vertices);
    triangles.resize(num_faces);


    // Read out the vertices
    for (int i=0; i<num_vertices; i++) {
      std::getline (off_file,line);
      std::vector<std::string> pts;
      boost::split(pts, line, [](char c){return c == ' ';});
      if (pts.size() != 3)
        std::invalid_argument("Vertex " + std::to_string(i) + " defines more or less than 3 points");
      pc->points[i].x = std::stof(pts[0]);
      pc->points[i].y = std::stof(pts[1]);
      pc->points[i].z = std::stof(pts[2]);
    }

    // Read out the faces
    for (int i=0; i<num_faces; i++) {
      std::getline (off_file,line);
      std::vector<std::string> face;
      boost::split(face, line, [](char c){return c == ' ';});
      if (face.size() != 4 || std::stoi(face[0]) != 3)
        std::invalid_argument("Face " + std::to_string(i) + " does not define a triangle");

      pcl::Vertices triangle;
      triangle.vertices.resize(3);
      triangle.vertices[0] = std::stoi(face[1]);
      triangle.vertices[1] = std::stoi(face[2]);
      triangle.vertices[2] = std::stoi(face[3]);
      triangles[i] = triangle;
    }

    while ( std::getline (off_file,line) )
    {
      std::cout << "WTF lines: " << line << std::endl;
    }
    off_file.close();
  }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void to_polygon_mesh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                     const std::vector<pcl::Vertices>& triangles,
                     pcl::PolygonMesh::Ptr mesh) {
  pcl::PCLPointCloud2 point_cloud2;
  pcl::toPCLPointCloud2(*pc, point_cloud2);

  mesh->cloud = point_cloud2;
  mesh->polygons = triangles;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

  ScopeTime t_total("Total computation", false);
  /****************************************************************************
  * Args processing
  ****************************************************************************/
  po::options_description desc("Generate depth map centered around the object in the scene\n");

  std::string input = "";
  std::string output = "./";
  bool viz = false;

  desc.add_options()
      ("help,h", "produce this help message")
      ("input,i", po::value<std::string>(&input)->default_value(input), "Mesh to render")
      ("output,o", po::value<std::string>(&output)->default_value(output), "Folder in which to save the point clouds")
      ("viz,v", po::value<bool>(&viz)->default_value(viz), "Enable viz");

  po::variables_map vm;
  po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
  std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
  po::store(parsed, vm);
  if (vm.count("help"))
  {
      std::cout << desc << std::endl;
      return false;
  }

  try {po::notify(vm);}
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false;
  }

  if ((output.size() > 0) && !(output[output.size()-1] == '/'))
      output += "/";


  /****************************************************************************
  * Setup
  ****************************************************************************/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::Vertices> triangles;

  readOFF(input, cloud, triangles);
  // std::cout << "   Pre procedure || Vertices: " << cloud->points.size() << " | Triangles: " << triangles.size() << std::endl;
  float scalefactor = 1.0f;
  scale_points_unit_sphere (*cloud, scalefactor);

  std::vector<std::vector<uint> > vertex_to_face;
  vertex_to_face.resize(cloud->points.size());
  for (uint i=0; i < vertex_to_face.size(); i++)
    vertex_to_face[i].reserve(5);

  for (uint i=0; i<triangles.size(); i++) {
    vertex_to_face[triangles[i].vertices[0]].push_back(i);
    vertex_to_face[triangles[i].vertices[1]].push_back(i);
    vertex_to_face[triangles[i].vertices[2]].push_back(i);
  }


  /****************************************************************************
  * Duplicates removal procedure
  ****************************************************************************/
  pcl::search::KdTree<pcl::PointXYZ> tree;
  tree.setInputCloud(cloud);

  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  float neigh_size = 0.001f;

  for (uint pt_idx=0; pt_idx<cloud->points.size(); pt_idx++) {
    if (std::isnan(cloud->points[pt_idx].x))
      continue;

    if (vertex_to_face[pt_idx].size() == 0) {
      cloud->points[pt_idx].x = cloud->points[pt_idx].y = cloud->points[pt_idx].z = std::numeric_limits<float>::quiet_NaN();
      continue;
    }


    tree.radiusSearch(cloud->points[pt_idx], neigh_size, k_indices, k_sqr_distances);
    // std::cout << "Pt " << pt_idx << " found " << k_indices.size() << " duplicates" << std::endl;

    for (uint k_idx=1; k_idx<k_indices.size(); k_idx++) {
      uint dup_idx = k_indices[k_idx];
      cloud->points[dup_idx].x = cloud->points[dup_idx].y = cloud->points[dup_idx].z = std::numeric_limits<float>::quiet_NaN();

      for (uint vtf_idx=0; vtf_idx<vertex_to_face[dup_idx].size(); vtf_idx++) {
        uint tri_idx = vertex_to_face[dup_idx][vtf_idx];
        if (triangles[tri_idx].vertices[0] == dup_idx)
          triangles[tri_idx].vertices[0] = pt_idx;
        else if (triangles[tri_idx].vertices[1] == dup_idx)
          triangles[tri_idx].vertices[1] = pt_idx;
        else if (triangles[tri_idx].vertices[2] == dup_idx)
          triangles[tri_idx].vertices[2] = pt_idx;
      }
    }
  }

  // Properly re-index triangles using indices
  std::vector<int> indices;
  cloud->is_dense = false;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  std::unordered_map<uint, uint> reverse_vertex_idx;
  for (uint i=0; i<indices.size(); i++) {
    reverse_vertex_idx[indices[i]] = i;
  }

  for (uint tri_idx=0; tri_idx < triangles.size(); tri_idx++) {
    triangles[tri_idx].vertices[0] = reverse_vertex_idx[triangles[tri_idx].vertices[0]];
    triangles[tri_idx].vertices[1] = reverse_vertex_idx[triangles[tri_idx].vertices[1]];
    triangles[tri_idx].vertices[2] = reverse_vertex_idx[triangles[tri_idx].vertices[2]];
  }

  // std::cout << "Post procedure || Vertices: " << cloud->points.size() << " | Triangles: " << triangles.size() << std::endl;


  /****************************************************************************
  * Saving and Viz
  ****************************************************************************/

  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
  to_polygon_mesh(cloud, triangles, mesh);


  std::string filename;
  for (uint i=input.size()-1; i > 0; i--)
  {
      if (input[i] == '/')
      {
          filename = input.substr(i+1, input.size() - i - 5);
          break;
      }
  }

  std::string save_filename = output + filename + ".ply";
  pcl::io::savePLYFileBinary (save_filename, *mesh);

  // Viz
  if (viz) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1., "coords", 0);
    viewer->addPolygonMesh (*mesh);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "cloud");

    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
    }
  }

  return 0;
}
