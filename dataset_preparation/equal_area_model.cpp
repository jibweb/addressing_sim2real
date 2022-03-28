#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <string>
#include <vector>

// #include <opencv2/opencv.hpp>
// #include <boost/filesystem.hpp>
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

std::string string_to_hex(const std::string& input)
{
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

void to_polygon_mesh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                     const std::vector<pcl::Vertices>& triangles,
                     pcl::PolygonMesh::Ptr mesh) {
  pcl::PCLPointCloud2 point_cloud2;
  pcl::toPCLPointCloud2(*pc, point_cloud2);

  mesh->cloud = point_cloud2;
  mesh->polygons = triangles;
}

float triangle_area(Eigen::Vector4f& p1, Eigen::Vector4f& p2, Eigen::Vector4f& p3) {
  float a,b,c,s;

  // Get the area of the triangle
  Eigen::Vector4f v21 (p2 - p1);
  Eigen::Vector4f v31 (p3 - p1);
  Eigen::Vector4f v23 (p2 - p3);
  a = v21.norm (); b = v31.norm (); c = v23.norm (); s = (a+b+c) * 0.5f + 1e-6;

  return sqrt(s * (s-a) * (s-b) * (s-c));
}

int
main(int argc, char* argv[]) {

  ScopeTime t_total("Total computation", false);
  /**************************************************************************
  * Args processing
  **************************************************************************/
  po::options_description desc("Generate depth map centered around the object in the scene\n");

  std::string input = "";
  std::string output = "./";
  float area = 0.1;

  desc.add_options()
      ("help,h", "produce this help message")
      ("input,i", po::value<std::string>(&input)->default_value(input), "Mesh to render")
      ("output,o", po::value<std::string>(&output)->default_value(output), "Folder in which to save the point clouds")
      ("area,a", po::value<float>(&area)->default_value(area), "Maximum area for a triangle");

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


  /**************************************************************************
  * Setup
  **************************************************************************/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::Vertices> triangles;

  readOFF(input, cloud, triangles);
  float scalefactor = 1.0f;
  // std::cout << cloud->points[0] << std::endl;
  double scale = scale_points_unit_sphere (*cloud, scalefactor);
  // std::cout << cloud->points[0] << std::endl;


  std::vector<float> face_area, cum_face_area;
  face_area.resize(triangles.size());
  cum_face_area.resize(triangles.size());
  float a,b,c,s;
  for (uint i=0; i < face_area.size(); i++) {
    Eigen::Vector4f p1 = cloud->points[triangles[i].vertices[0]].getVector4fMap ();
    Eigen::Vector4f p2 = cloud->points[triangles[i].vertices[1]].getVector4fMap ();
    Eigen::Vector4f p3 = cloud->points[triangles[i].vertices[2]].getVector4fMap ();

    // Get the area of the triangle
    // Eigen::Vector4f v21 (p2 - p1);
    // Eigen::Vector4f v31 (p3 - p1);
    // Eigen::Vector4f v23 (p2 - p3);
    // a = v21.norm (); b = v31.norm (); c = v23.norm (); s = (a+b+c) * 0.5f + 1e-6;

    // face_area[i] = sqrt(s * (s-a) * (s-b) * (s-c));
    face_area[i] = triangle_area(p1, p2, p3);

    if (std::isnan(face_area[i])) {
      // std::cout << "area is nan: " << s << " " << a << " " << b << " " << c << std::endl;
    }

    if (i==0)
      cum_face_area[i] = face_area[i];
    else
      cum_face_area[i] = cum_face_area[i-1] + face_area[i];
  }

  for (uint i=0; i<10; i++) {
    std::cout << face_area[i] << " " << cum_face_area[i] << std::endl;
  }

  double avg_area = cum_face_area[cum_face_area.size() - 1] / cum_face_area.size();
  std::cout << "Average face area: " << avg_area << std::endl;

  for (uint i=0; i<triangles.size(); i++) {
    if (face_area[i] < 2*avg_area)
      continue;

    uint p0_idx = triangles[i].vertices[0];
    uint p1_idx = triangles[i].vertices[1];
    uint p2_idx = triangles[i].vertices[2];

    Eigen::Vector4f p1 = cloud->points[p0_idx].getVector4fMap ();
    Eigen::Vector4f p2 = cloud->points[p1_idx].getVector4fMap ();
    Eigen::Vector4f p3 = cloud->points[p2_idx].getVector4fMap ();

    Eigen::Vector4f barycenter = (p1 + p2 + p3) / 3.;
    uint new_pt_idx = cloud->points.size();
    pcl::PointXYZ p;
    p.x = barycenter(0);
    p.y = barycenter(1);
    p.z = barycenter(2);
    cloud->points.push_back(p);

    pcl::Vertices tri1;
    tri1.vertices.push_back(p0_idx);
    tri1.vertices.push_back(p1_idx);
    tri1.vertices.push_back(new_pt_idx);
    triangles[i] = tri1;
    face_area[i] = triangle_area(p1, p2, barycenter);

    pcl::Vertices tri2;
    tri2.vertices.push_back(p1_idx);
    tri2.vertices.push_back(p2_idx);
    tri2.vertices.push_back(new_pt_idx);
    triangles.push_back(tri2);
    face_area.push_back(triangle_area(p2, p3, barycenter));

    pcl::Vertices tri3;
    tri3.vertices.push_back(p2_idx);
    tri3.vertices.push_back(p0_idx);
    tri3.vertices.push_back(new_pt_idx);
    triangles.push_back(tri3);
    face_area.push_back(triangle_area(p3, p1, barycenter));

    i--;
  }

  std::cout << "This mesh has " << triangles.size() << " triangles" << std::endl;


  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
  to_polygon_mesh(cloud, triangles, mesh);

  // pcl::io::savePLYFileBinary ("test.ply", *mesh);

  // Viz
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1., "coords", 0);
  viewer->addPolygonMesh (*mesh);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "cloud");

  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }


  return 0;
}
