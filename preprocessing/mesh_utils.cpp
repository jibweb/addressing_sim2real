#include <vector>


float edgeFunction(const double &a0, const double &a1,
                   const double &b0, const double &b1,
                   const double &c0, const double &c1) {
 return (c0 - a0) * (b1 - a1) - (c1 - a1) * (b0 - a0);
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


void bresenham_line_low(int x0, int y0, int x1, int y1, std::vector<int> & min_y, std::vector<int> & max_y, int min_x) {
  int dx = x1 - x0;
  int dy = y1 - y0;
  int yi = 1;
  if (dy < 0) {
    yi = -1;
    dy = -dy;
  }

  int D = 2*dy - dx;
  int y = y0;

  for (int x= x0; x<=x1; x++) {
    // "plot" the x,y value
    int x_idx = x - min_x;
    if (y > max_y[x_idx])
      max_y[x_idx] = y;

    if (y < min_y[x_idx])
      min_y[x_idx] = y;

    if (D>0) {
      y += yi;
      D -= 2*dx;
    }
    D += 2*dy;
  }
}


void bresenham_line_high(int x0, int y0, int x1, int y1, std::vector<int> & min_y, std::vector<int> & max_y, int min_x) {
  int dx = x1 - x0;
  int dy = y1 - y0;
  int xi = 1;

  if (dx < 0) {
    xi = -1;
    dx = -dx;
  }

  int D = 2*dx - dy;
  int x = x0;

  for (int y=y0; y<=y1; y++) {
    // "plot" the x,y value
    int x_idx = x - min_x;
    if (y > max_y[x_idx])
      max_y[x_idx] = y;

    if (y < min_y[x_idx])
      min_y[x_idx] = y;
    if (D>0) {
      x += xi;
      D -= 2*dy;
    }
    D += 2*dx;
  }
}

void bresenham_line(int x0, int y0, int x1, int y1, std::vector<int> & min_y, std::vector<int> & max_y, int min_x) {
  if (abs(y1 - y0) < abs(x1 - x0)) {
    if (x0 > x1) {
      bresenham_line_low(x1, y1, x0, y0, min_y, max_y, min_x);
    } else {
      bresenham_line_low(x0, y0, x1, y1, min_y, max_y, min_x);
    }
  } else {
    if (y0 > y1) {
      bresenham_line_high(x1, y1, x0, y0, min_y, max_y, min_x);
    } else {
      bresenham_line_high(x0, y0, x1, y1, min_y, max_y, min_x);
    }
  }
}


void rasterize(const Eigen::MatrixXi & F,
               const Eigen::MatrixXd & V_uv,
               const Eigen::VectorXd & V_z,
               Eigen::MatrixXd & W0,
               Eigen::MatrixXd & W1,
               Eigen::MatrixXd & W2,
               Eigen::MatrixXi & I_face_idx,
               Eigen::MatrixXd & image_mask,
               const uint image_size,
               const double min_px,
               const double max_px) {

  Eigen::MatrixXd Z_buffer = Eigen::MatrixXd::Constant(image_size+1, image_size+1, -1.e3);


  // --- Rasterization ----------------------------------------------------
  for (uint face_idx=0; face_idx<F.rows(); face_idx++) {
    int min_x = image_size + 1;
    int max_x = -1;
    std::vector<int> vx(3);
    std::vector<int> vy(3);

    if (V_uv(F(face_idx, 0), 0) < min_px || V_uv(F(face_idx, 0), 0) > max_px ||
        V_uv(F(face_idx, 0), 1) < min_px || V_uv(F(face_idx, 0), 1) > max_px ||
        V_uv(F(face_idx, 1), 0) < min_px || V_uv(F(face_idx, 1), 0) > max_px ||
        V_uv(F(face_idx, 1), 1) < min_px || V_uv(F(face_idx, 1), 1) > max_px ||
        V_uv(F(face_idx, 2), 0) < min_px || V_uv(F(face_idx, 2), 0) > max_px ||
        V_uv(F(face_idx, 2), 1) < min_px || V_uv(F(face_idx, 2), 1) > max_px)
        continue;

    // Get the pixel boundaries of the triangle
    for (uint i=0; i<3; i++) {
      vx[i] = image_size * (V_uv(F(face_idx, i), 0) - min_px) / (max_px - min_px);
      vy[i] = image_size * (V_uv(F(face_idx, i), 1) - min_px) / (max_px - min_px);

      if (vx[i] < min_x)
        min_x = vx[i];

      if (vx[i] > max_x)
        max_x = vx[i];
    }

    // Get the range of y for each x in range of this triangle
    // aka fully define the area where the triangle needs to be drawn
    std::vector<int> max_y(max_x-min_x+1, -1);
    std::vector<int> min_y(max_x-min_x+1, image_size + 1);

    for(uint i=0; i<3; i++) {
      bresenham_line(vx[i], vy[i], vx[(i+1)%3], vy[(i+1)%3], min_y, max_y, min_x);
    }

    // Once we have the boundaries of the triangles, draw it !
    float tri_area = abs((vx[2] - vx[0])*(vy[1] - vy[0]) - (vy[2] - vy[0])*(vx[1] - vx[0])); //Twice the area but who cares

    if (tri_area == 0.)
      continue;

    for (uint i=0; i<max_y.size(); i++) {
      // Compute the barycentric coordinates and the step update
      float w0 = (min_x + static_cast<int>(i) - vx[1]) * (vy[2] - vy[1]) - (min_y[i] - vy[1]) * (vx[2] - vx[1]);
      float w1 = (min_x + static_cast<int>(i) - vx[2]) * (vy[0] - vy[2]) - (min_y[i] - vy[2]) * (vx[0] - vx[2]);
      float w2 = (min_x + static_cast<int>(i) - vx[0]) * (vy[1] - vy[0]) - (min_y[i] - vy[0]) * (vx[1] - vx[0]);

      w0 /= tri_area;
      w1 /= tri_area;
      w2 /= tri_area;

      if (std::isnan(w0) || std::isnan(w1) || std::isnan(w2)) {
        std::cout << "w: "<< w0 << " " << w1 << " " << w2 << " " << tri_area << std::endl;
      }

      float w0_stepy, w1_stepy, w2_stepy;

      w0_stepy = -(vx[2] - vx[1]) / tri_area;
      w1_stepy = -(vx[0] - vx[2]) / tri_area;
      w2_stepy = -(vx[1] - vx[0]) / tri_area;

      if (std::isnan(w0_stepy) || std::isnan(w1_stepy) || std::isnan(w2_stepy)) {
        std::cout << "w_step: " << w0_stepy << " " << w1_stepy << " " << w2_stepy << " " << tri_area << std::endl;
      }

      for (uint j=min_y[i]; j<max_y[i]; j++) {
        double z_interp = w0 * V_z(F(face_idx, 0))
                        + w1 * V_z(F(face_idx, 1))
                        + w2 * V_z(F(face_idx, 2));
        if (z_interp > Z_buffer((min_x + i), j)) {
          image_mask((min_x + i), j) = 1.;

          W0((min_x + i), j) = (w0);
          W1((min_x + i), j) = (w1);
          W2((min_x + i), j) = (w2);
          I_face_idx((min_x + i), j) = face_idx;

          Z_buffer((min_x + i), j) = z_interp;
        }

        w0 += w0_stepy;
        w1 += w1_stepy;
        w2 += w2_stepy;
      }
    }
  } // for loop on faces
}
