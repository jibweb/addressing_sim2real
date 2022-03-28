#include <string>

#include <boost/program_options.hpp>
#include <pcl/conversions.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>

namespace po = boost::program_options;


///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

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
  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
  // Read the mesh
  if (pcl::io::loadPLYFile(input.c_str(), *mesh) == -1) {
    PCL_ERROR("Couldn't read %s file \n", input.c_str());
    return -1;
  }

  /****************************************************************************
  * Saving and Viz
  ****************************************************************************/


  std::string filename;
  for (uint i=input.size()-1; i > 0; i--)
  {
      if (input[i] == '/')
      {
          filename = input.substr(i+1, input.size() - i - 5);
          break;
      }
  }

  std::string save_filename = output + filename + "_bin.ply";
  pcl::io::savePLYFileBinary (save_filename, *mesh);

  // Viz
  if (viz) {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1., "coords", 0);
    viewer->addPolygonMesh (*mesh);

    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
    }
  }

  return 0;
}
