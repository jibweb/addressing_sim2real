#include <vector>


float edgeFunction(const double &a0, const double &a1,
                   const double &b0, const double &b1,
                   const double &c0, const double &c1) {
 return (c0 - a0) * (b1 - a1) - (c1 - a1) * (b0 - a0);
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

  // coords_x.reserve(x1-x0+1);
  // coords_y.reserve(x1-x0+1);

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

  // coords_x.reserve(y1-y0+1);
  // coords_y.reserve(y1-y0+1);

  for (int y=y0; y<=y1; y++) {
    // "plot" the x,y value
    int x_idx = x - min_x;
    if (y > max_y[x_idx])
      max_y[x_idx] = y;

    if (y < min_y[x_idx])
      min_y[x_idx] = y;
    // coords_x.push_back(x);
    // coords_y.push_back(y);
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

// void bresenham_line_low(int x0, int y0, int x1, int y1, std::vector<int> & coords_x, std::vector<int> & coords_y) {
//   int dx = x1 - x0;
//   int dy = y1 - y0;
//   int yi = 1;
//   if (dy < 0) {
//     yi = -1;
//     dy = -dy;
//   }

//   int D = 2*dy - dx;
//   int y = y0;

//   coords_x.reserve(x1-x0+1);
//   coords_y.reserve(x1-x0+1);

//   for (int x= x0; x<=x1; x++) {
//     // "plot" the x,y value
//     coords_x.push_back(x);
//     coords_y.push_back(y);
//     if (D>0) {
//       y += yi;
//       D -= 2*dx;
//     }
//     D += 2*dy;
//   }
// }


// void bresenham_line_high(int x0, int y0, int x1, int y1, std::vector<int> & coords_x, std::vector<int> & coords_y) {
//   int dx = x1 - x0;
//   int dy = y1 - y0;
//   int xi = 1;

//   if (dx < 0) {
//     xi = -1;
//     dx = -dx;
//   }

//   int D = 2*dx - dy;
//   int x = x0;

//   coords_x.reserve(y1-y0+1);
//   coords_y.reserve(y1-y0+1);

//   for (int y=y0; y<=y1; y++) {
//     // "plot" the x,y value
//     coords_x.push_back(x);
//     coords_y.push_back(y);
//     if (D>0) {
//       x += xi;
//       D -= 2*dy;
//     }
//     D += 2*dx;
//   }
// }

// void bresenham_line(int x0, int y0, int x1, int y1, std::vector<int> & coords_x, std::vector<int> & coords_y) {
//   if (abs(y1 - y0) < abs(x1 - x0)) {
//     if (x0 > x1) {
//       bresenham_line_low(x1, y1, x0, y0, coords_x, coords_y);
//     } else {
//       bresenham_line_low(x0, y0, x1, y1, coords_x, coords_y);
//     }
//   } else {
//     if (y0 > y1) {
//       bresenham_line_high(x1, y1, x0, y0, coords_x, coords_y);
//     } else {
//       bresenham_line_high(x0, y0, x1, y1, coords_x, coords_y);
//     }
//   }
// }
