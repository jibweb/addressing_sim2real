// This file exists to create a nice static or shared library via cmake
// but can otherwise be omitted if you prefer to compile tinyply
// directly into your own project.
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <fstream>

#include <pcl/PolygonMesh.h>
#include <pcl/common/projection_matrix.h>

namespace tinyply {
  union Float
  {
      float f;
      char s[sizeof(float)];
  };

  union Uint
  {
      uint32_t f;
      char s[sizeof(uint32_t)];
  };

  template<class PointT>
  void loadPLY(const std::string filepath,
               pcl::PointCloud<PointT> & pc,
               pcl::PolygonMesh & mesh) {

    std::ifstream ss(filepath, std::ios::binary);
    if (ss.fail()) throw std::runtime_error("failed to open " + filepath);

    PlyFile file;
    file.parse_header(ss);

    // Tinyply treats parsed data as untyped byte buffers. See below for examples.
    std::shared_ptr<PlyData> ply_verts, ply_norms, ply_faces;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the header prior to reading the data. For brevity of this sample, properties
    // like vertex position are hard-coded:
    try { ply_verts = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { ply_norms = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    // Providing a list size hint (the last argument) is a 2x performance improvement. If you have
    // arbitrary ply files, it is best to leave this 0.
    try { ply_faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


    file.read(ss);

    // Copying the buffer into the point cloud (both coordinates and normals)
    uint8_t * vert_buff_ptr = ply_verts->buffer.get();
    uint8_t * norm_buff_ptr = ply_norms->buffer.get();
    pc.points.resize(ply_verts->count);
    for (uint pt_idx=0; pt_idx<ply_verts->count; pt_idx++) {
      for (uint i=0; i<3; i++) {
        Float vert_f, norm_f;
        vert_f.s[0] = *vert_buff_ptr++;
        vert_f.s[1] = *vert_buff_ptr++;
        vert_f.s[2] = *vert_buff_ptr++;
        vert_f.s[3] = *vert_buff_ptr++;

        norm_f.s[0] = *norm_buff_ptr++;
        norm_f.s[1] = *norm_buff_ptr++;
        norm_f.s[2] = *norm_buff_ptr++;
        norm_f.s[3] = *norm_buff_ptr++;

        pc.points[pt_idx].data[i] = vert_f.f;
        pc.points[pt_idx].normal[i] = norm_f.f;
      }
    }

    // Copying the triangles in the pcl::PoylgonMesh
    uint8_t * face_buff_ptr = ply_faces->buffer.get();
    mesh.polygons.resize(ply_faces->count);
    for (uint tri_idx=0; tri_idx<ply_faces->count; tri_idx++) {
      mesh.polygons[tri_idx].vertices.resize(3);
      for (uint i=0; i<3; i++) {
        Uint f;
        f.s[0] = *face_buff_ptr++;
        f.s[1] = *face_buff_ptr++;
        f.s[2] = *face_buff_ptr++;
        f.s[3] = *face_buff_ptr++;

        mesh.polygons[tri_idx].vertices[i] = f.f;
      }
    }
  }
}
