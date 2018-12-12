#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>
// #include <v4r/common/camera.h>
// #include <v4r/common/pcl_opencv.h>
#include <v4r/rendering/depthmapRenderer.h>
#include <v4r/rendering/dmRenderObject.h>

#include "scope_time.h"

namespace po = boost::program_options;

int
main(int argc, char* argv[]) {

  ScopeTime t_total("Total computation", false);
  /**************************************************************************
  * Args processing
  **************************************************************************/
  po::options_description desc("Generate depth map centered around the object in the scene\n");

  std::string input = "";
  std::string output = "./";
  float sphere_distance = 3.f;
  bool visualize = false;
  uint pose_per_traj = 60;

  desc.add_options()
      ("help,h", "produce this help message")
      ("input,i", po::value<std::string>(&input)->default_value(input), "Mesh to render")
      ("output,o", po::value<std::string>(&output)->default_value(output), "Folder in which to save the point clouds")
      ("sphere_distance,d", po::value<float>(&sphere_distance)->default_value(sphere_distance), "Distance to the object sphere when rendering")
      ("visualize,v", po::bool_switch(&visualize), "visualize results");

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
  // CAD Rendering pipeline
  v4r::DepthmapRendererModel model = v4r::DepthmapRendererModel(input);
  v4r::DepthmapRenderer dmr = v4r::DepthmapRenderer(640, 480);
  std::vector<Eigen::Vector3f> sphere_positions = dmr.createSphere(sphere_distance, 0);
  dmr.setIntrinsics(567.6, 570.2, 324.7, 250.1); // cf def in Freiburg 3 in https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  dmr.setModel(&model);

  // z vector for trajectory estimation
  Eigen::Vector3f z;
  z << 0., 0., 1.;

  // Image saving options setup
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  for (uint sp = 0; sp < sphere_positions.size(); sp++) {
    // Creating directory
    std::string track_dir = output + "track_" + std::to_string(sp) + "/";
    boost::filesystem::path dir(track_dir.c_str());
    if(!boost::filesystem::create_directory(dir)) {
      std::cerr<< "Error creating directory "<< track_dir <<std::endl;
    }

    // Creating directory for images
    boost::filesystem::path dir_depth((track_dir + "depth/").c_str());
    boost::filesystem::create_directory(dir_depth);

    cv::Mat rgb(480, 640, CV_8UC3);
    rgb.setTo(cv::Scalar(255, 255, 255));
    try {
      cv::imwrite(track_dir + "rgb.png", rgb, compression_params);
    }
    catch (std::runtime_error& ex) {
      fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
      return 1;
    }

    // Creating association file and groundtruth
    std::ofstream association;
    association.open(track_dir + "associations.txt");
    for (uint i=0; i < pose_per_traj; i++) {
      association << std::to_string(static_cast<float>(i))
                  << " depth/" + std::to_string(i) + ".png "
                  << std::to_string(static_cast<float>(i))
                  << " rgb.png"
                  << std::endl;
    }
    association.close();

    std::ofstream groundtruth;
    groundtruth.open(track_dir + "groundtruth.txt");

    /**************************************************************************
    * Compute camera pose
    **************************************************************************/
    Eigen::Vector3f position = sphere_positions[sp];
    Eigen::Vector3f n_pos;
    n_pos(0) = position(0);
    n_pos(1) = position(1);
    n_pos(2) = position(2);
    n_pos.normalize();

    if (acos(fabs(n_pos.dot(z))) < 20*M_PI/180.) {
      continue;
    }

    Eigen::Vector3f traj_vec = n_pos.cross(z);
    traj_vec.normalize();

    for (uint pos_idx=0; pos_idx < pose_per_traj; pos_idx++) {
      Eigen::Vector3f real_pos = position + (static_cast<float>(pos_idx) - 30.)/15.*traj_vec;
      Eigen::Matrix4f pose = dmr.getPoseLookingToCenterFrom(real_pos);
      dmr.setCamPose(pose);

      Eigen::Matrix3f rotation = pose.block(0, 0, 3, 3);
      rotation = rotation.transpose();
      Eigen::Quaternionf q(rotation);
      groundtruth << std::to_string(static_cast<float>(pos_idx)) << " "
                  << real_pos(0) << " " << real_pos(1) << " " << real_pos(2) << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                  << std::endl;

      /******************************************************************
      * Rendering of a depthmap
      ******************************************************************/
      cv::Mat depth_img, color, normal;
      float viz_surf_area;
      depth_img = dmr.renderDepthmap(viz_surf_area, color, normal);

      // Transform it into a xtion-like depthmap
      cv::Mat depth_img_xtion =  cv::Mat_<uint16_t>(480, 640);
      depth_img_xtion.setTo(static_cast<uint16_t>(0));
      for (uint h=0; h < 480; h++)
          for(uint w=0; w < 640 ; w++)
              depth_img_xtion.at<uint16_t>(h, w) = static_cast<uint16_t>(5000. * depth_img.at<float>(h, w));

      if (visualize)
      {
        cv::imshow("Depthmap", depth_img_xtion);
        cv::waitKey(0);
      }

      // Saving the image
      std::string depth_img_fn = track_dir + "depth/" + std::to_string(pos_idx) + ".png";
      try {
        cv::imwrite(depth_img_fn, depth_img_xtion, compression_params);
      }
      catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 1;
      }
    }

    groundtruth.close();
  }

  return 0;
}
