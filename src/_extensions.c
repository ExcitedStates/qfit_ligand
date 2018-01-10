#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"

#define SQUARE(x) ((x) * (x))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define NEAREST_INT(a) ((int) round(a))


static int modulo(int x, int N)
{
  int ret = x % N;
  if (ret < 0)
    ret += N;
  return ret;
}


static PyObject *correlation_gradients(PyObject *dummy, PyObject *args)
{
  // Parse arguments
  PyObject *py_coor_arg=NULL, *py_occ_arg=NULL, *py_lmax_arg=NULL, *py_derivatives_arg=NULL,
           *py_grid_to_cart_arg=NULL, *py_target_arg=NULL, *py_gradients_arg=NULL;
  double rstep, rmax;
  PyArrayObject *py_coor=NULL, *py_occ=NULL, *py_lmax=NULL,
                *py_derivatives=NULL, *py_grid_to_cart=NULL, *py_target=NULL,
                *py_gradients=NULL;

  if (!PyArg_ParseTuple(args, "OOOOddOOO",
              &py_coor_arg, &py_occ_arg, &py_lmax_arg, &py_derivatives_arg, &rstep, &rmax,
              &py_grid_to_cart_arg, &py_target_arg, &py_gradients_arg))
      return NULL;

  py_coor = (PyArrayObject *)
      PyArray_FROM_OTF(py_coor_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_coor == NULL)
      goto fail;

  py_occ = (PyArrayObject *)
      PyArray_FROM_OTF(py_occ_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_occ == NULL)
      goto fail;

  py_lmax = (PyArrayObject *)
      PyArray_FROM_OTF(py_lmax_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_lmax == NULL)
      goto fail;

  py_derivatives = (PyArrayObject *)
      PyArray_FROM_OTF(py_derivatives_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_derivatives == NULL)
      goto fail;

  py_grid_to_cart = (PyArrayObject *)
      PyArray_FROM_OTF(py_grid_to_cart_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_grid_to_cart == NULL)
      goto fail;

  py_target = (PyArrayObject *)
      PyArray_FROM_OTF(py_target_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_target == NULL)
      goto fail;

  py_gradients = (PyArrayObject *)
      PyArray_FROM_OTF(py_gradients_arg, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);
  if (py_gradients == NULL)
      goto fail;

  // Get pointers to arrays and shape info.
  double *coor = (double *) PyArray_DATA(py_coor);
  double *derivatives = (double *) PyArray_DATA(py_derivatives);
  double *lmax = (double *) PyArray_DATA(py_lmax);
  double *occ = (double *) PyArray_DATA(py_occ);
  double *grid_to_cart = (double *) PyArray_DATA(py_grid_to_cart);
  double *target = (double *) PyArray_DATA(py_target);
  double *gradients = (double *) PyArray_DATA(py_gradients);

  npy_intp *coor_shape = PyArray_DIMS(py_coor);
  npy_intp *derivatives_shape = PyArray_DIMS(py_derivatives);
  npy_intp *target_shape = PyArray_DIMS(py_target);

    // After the parsing and generation of pointers, the real function can
    // Precompute some values.
    int target_slice = target_shape[2] * target_shape[1];
    int target_size = target_slice * target_shape[0];
    double rmax2 = rmax * rmax;

    // Loop over coordinates
    int n;
    for (n = 0; n < coor_shape[0]; n++) {
        double q = occ[n];
        if (q == 0.0) {
            continue;
        }

        int coor_ind = 3 * n;
        double center_a = coor[coor_ind];
        double center_b = coor[coor_ind + 1];
        double center_c = coor[coor_ind + 2];

        int cmin = (int) ceil(center_c - lmax[2]);
        int bmin = (int) ceil(center_b - lmax[1]);
        int amin = (int) ceil(center_a - lmax[0]);

        int cmax = (int) floor(center_c + lmax[2]);
        int bmax = (int) floor(center_b + lmax[1]);
        int amax = (int) floor(center_a + lmax[0]);

        int derivatives_ind = n * derivatives_shape[1];

        int c;
        for (c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * target_slice, target_size);
            double dc = center_c - c;
            double dz = grid_to_cart[8] * dc;
            double dy_c = grid_to_cart[5] * dc;
            double dx_c = grid_to_cart[2] * dc;

            double gradient_z = q * derivatives[derivatives_ind + NEAREST_INT(fabs(dz) / rstep)];
            if (dz < 0) {
                gradient_z *= -1;
            }
            double dz2 = dz * dz;

            int b;
            for (b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * target_shape[2], target_slice) + ind_c;
                double db = center_b - b;
                double dy = dy_c + grid_to_cart[4] * db;
                double dx_cb = dx_c + grid_to_cart[1] * db;

                double gradient_y = q * derivatives[derivatives_ind + NEAREST_INT(fabs(dy) / rstep)];
                if (dy < 0) {
                    gradient_y *= -1;
                }
                double dz2_dy2 = dz2 + dy * dy;

                int a;
                for (a = amin; a <= amax; a++) {
                    double da = center_a - a;
                    double dx = dx_cb + grid_to_cart[0] * da;
                    double dz2_dy2_dx2 = dz2_dy2 + dx * dx;
                    if (dz2_dy2_dx2 > rmax2) {
                        continue;
                    }
                    int ind = ind_cb + modulo(a, target_shape[2]);
                    double density = target[ind];
                    double gradient_x = q * derivatives[derivatives_ind + NEAREST_INT(fabs(dx) / rstep)];
                    if (dx < 0) {
                        gradient_x *= -1;
                    }
                    gradients[coor_ind] += density * gradient_x;
                    gradients[coor_ind + 1] += density * gradient_y;
                    gradients[coor_ind + 2] += density * gradient_z;
                }
            }
        }
    }


  // Clean up objects
  Py_DECREF(py_coor);
  Py_DECREF(py_occ);
  Py_DECREF(py_lmax);
  Py_DECREF(py_derivatives);
  Py_DECREF(py_grid_to_cart);
  Py_DECREF(py_target);
  Py_DECREF(py_gradients);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  // Clean up objects
  Py_XDECREF(py_coor);
  Py_XDECREF(py_occ);
  Py_XDECREF(py_lmax);
  Py_XDECREF(py_derivatives);
  Py_XDECREF(py_grid_to_cart);
  Py_XDECREF(py_target);
  PyArray_DiscardWritebackIfCopy(py_gradients);
  Py_XDECREF(py_gradients);
  return NULL;
}


static PyObject *py_extend_to_p1(PyObject *dummy, PyObject *args)
{
  PyObject *py_grid_arg, *py_offset_arg, *py_symop_arg, *py_out_arg;
  PyArrayObject *py_grid=NULL, *py_offset=NULL, *py_symop=NULL, *py_out=NULL;

  if (!PyArg_ParseTuple(args, "OOOO", &py_grid_arg, &py_offset_arg, &py_symop_arg, &py_out_arg))
    return NULL;

  py_grid = (PyArrayObject *) PyArray_FROM_OTF(py_grid_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_grid == NULL)
    goto fail;
  py_offset = (PyArrayObject *) PyArray_FROM_OTF(py_offset_arg, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  if (py_offset == NULL)
    goto fail;
  py_symop = (PyArrayObject *) PyArray_FROM_OTF(py_symop_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
  if (py_symop == NULL)
    goto fail;
  py_out = (PyArrayObject *) PyArray_FROM_OTF(py_out_arg, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);
  if (py_out == NULL)
    goto fail;

  // Get pointers to arrays and shape info.
  double *grid = (double *) PyArray_DATA(py_grid);
  int *offset = (int *) PyArray_DATA(py_offset);
  double *symop = (double *) PyArray_DATA(py_symop);
  double *out = (double *) PyArray_DATA(py_out);

  npy_intp *grid_shape = PyArray_DIMS(py_grid);
  npy_intp *out_shape = PyArray_DIMS(py_out);
  int out_slice = out_shape[2] * out_shape[1];
  int grid_slice = grid_shape[2] * grid_shape[1];

  // Loop over all grid points and transform them to out points
  int z;
  for (z = 0; z < grid_shape[0]; ++z) {
    int grid_z = z + offset[2];
    int out_z_z = (int) (symop[11] + symop[10] * grid_z);
    int out_y_z = (int) (symop[7]  + symop[6]  * grid_z);
    int out_x_z = (int) (symop[3]  + symop[2]  * grid_z);
    int grid_index_z = z * grid_slice;
    int y;
    for (y = 0; y < grid_shape[1]; ++y) {
      int grid_y = y + offset[1];
      int out_z_zy = (int) (out_z_z + symop[9] * grid_y);
      int out_y_zy = (int) (out_y_z + symop[5] * grid_y);
      int out_x_zy = (int) (out_x_z + symop[1] * grid_y);
      int grid_index_zy = grid_index_z + y * grid_shape[2];
      int x;
      for (x = 0; x < grid_shape[2]; ++x) {
        int grid_x = x + offset[0];
        int out_z = (int) (out_z_zy + symop[8] * grid_x);
        int out_y = (int) (out_y_zy + symop[4] * grid_x);
        int out_x = (int) (out_x_zy + symop[0] * grid_x);
        int out_index = modulo(out_z, out_shape[0]) * out_slice +
                        modulo(out_y, out_shape[1]) * out_shape[2] +
                        modulo(out_x, out_shape[2]);
        int grid_index = grid_index_zy + x;
        out[out_index] = grid[grid_index];
      }
    }
  }
  // Clean up objects
  Py_DECREF(py_grid);
  Py_DECREF(py_offset);
  Py_DECREF(py_symop);
  Py_DECREF(py_out);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  // Clean up objects
  Py_XDECREF(py_grid);
  Py_XDECREF(py_offset);
  Py_XDECREF(py_symop);
  PyArray_DiscardWritebackIfCopy(py_out);
  Py_XDECREF(py_out);
  return NULL;
}

static PyObject *dilate_points(PyObject *dummy, PyObject *args)
{
    // Parse arguments
    PyObject
        *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL, *arg7=NULL, *arg8=NULL;
    double
        rstep, rmax;
    PyArrayObject *py_points=NULL, *py_occupancies=NULL, *py_lmax=NULL,
                  *py_radial_densities=NULL, *py_grid_to_cartesian=NULL,
                  *py_out=NULL;

    if (!PyArg_ParseTuple(args, "OOOOddOO",
                &arg1, &arg2, &arg3, &arg4, &rstep, &rmax, &arg7, &arg8))
        return NULL;

    py_points = (PyArrayObject *)
        PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_occupancies = (PyArrayObject *)
        PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_lmax = (PyArrayObject *)
        PyArray_FROM_OTF(arg3, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_lmax == NULL)
        goto fail;

    py_radial_densities = (PyArrayObject *)
        PyArray_FROM_OTF(arg4, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_radial_densities == NULL)
        goto fail;

    py_grid_to_cartesian = (PyArrayObject *)
        PyArray_FROM_OTF(arg7, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_grid_to_cartesian == NULL)
        goto fail;

    py_out = (PyArrayObject *)
        PyArray_FROM_OTF(arg8, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);
    if (py_out == NULL)
        goto fail;

    // Get pointers to arrays and shape info.
    double *points = (double *) PyArray_DATA(py_points);
    double *radial_densities = (double *) PyArray_DATA(py_radial_densities);
    double *lmax = (double *) PyArray_DATA(py_lmax);
    double *occupancies = (double *) PyArray_DATA(py_occupancies);
    double *grid_to_cartesian = (double *) PyArray_DATA(py_grid_to_cartesian);
    double *out = (double *) PyArray_DATA(py_out);

    npy_intp *points_shape = PyArray_DIMS(py_points);
    npy_intp *radial_densities_shape = PyArray_DIMS(py_radial_densities);
    npy_intp *out_shape = PyArray_DIMS(py_out);

    // After the parsing and generation of pointers, the real function can
    // start.

    // Precompute some values.
    int out_slice = out_shape[2] * out_shape[1];
    int out_size = out_slice * out_shape[0];
    double rmax2 = SQUARE(rmax);

    int n;
    for (n = 0; n < points_shape[0]; n++) {
        double q = occupancies[n];
        if (q == 0) {
            continue;
        }

        int points_ind = 3 * n;
        int radial_densities_ind = n * radial_densities_shape[1];
        double center_a = points[points_ind];
        double center_b = points[points_ind + 1];
        double center_c = points[points_ind + 2];

        int cmin = (int) ceil(center_c - lmax[2]);
        int bmin = (int) ceil(center_b - lmax[1]);
        int amin = (int) ceil(center_a - lmax[0]);

        int cmax = (int) floor(center_c + lmax[2]);
        int bmax = (int) floor(center_b + lmax[1]);
        int amax = (int) floor(center_a + lmax[0]);

        int c;
        for (c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * out_slice, out_size);
            double dc = center_c - c;
            double dist2_z = SQUARE(grid_to_cartesian[8] * dc);
            double disty_c = grid_to_cartesian[5] * dc;
            double distx_c = grid_to_cartesian[2] * dc;

            int b;
            for (b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * out_shape[2], out_slice) + ind_c;
                double db = center_b - b;
                double dist2_zy = dist2_z +
                    SQUARE(disty_c + grid_to_cartesian[4] * db);
                double distx_cb = distx_c + grid_to_cartesian[1] * db;

                int a;
                for (a = amin; a <= amax; a++) {
                    double da = center_a - a;
                    double dist2_zyx = dist2_zy +
                        SQUARE(distx_cb + grid_to_cartesian[0] * da);
                    if (dist2_zyx <= rmax2) {
                        double r = sqrt(dist2_zyx);
                        int index =
                            radial_densities_ind + NEAREST_INT(r / rstep);
                        out[ind_cb + modulo(a, out_shape[2])] +=
                            q * radial_densities[index];
                    }
                }
            }
        }
    }

    // Clean up objects
    Py_DECREF(py_points);
    Py_DECREF(py_occupancies);
    Py_DECREF(py_lmax);
    Py_DECREF(py_radial_densities);
    Py_DECREF(py_grid_to_cartesian);
    Py_DECREF(py_out);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    // Clean up objects
    Py_XDECREF(py_points);
    Py_XDECREF(py_radial_densities);
    Py_XDECREF(py_lmax);
    Py_XDECREF(py_occupancies);
    Py_XDECREF(py_grid_to_cartesian);
    PyArray_DiscardWritebackIfCopy(py_out);
    Py_XDECREF(py_out);
    return NULL;
}


static PyObject *mask_points(PyObject *dummy, PyObject *args)
{

    // Parse arguments
    PyObject
        *arg1=NULL, *arg2=NULL, *arg3, *arg5=NULL, *arg7=NULL;
    PyArrayObject
        *py_points=NULL, *py_occupancies=NULL, *py_lmax=NULL,
        *py_grid_to_cartesian=NULL, *py_out=NULL;
    double
        rmax, value;

    if (!PyArg_ParseTuple(args, "OOOdOdO",
                &arg1, &arg2, &arg3, &rmax, &arg5, &value, &arg7))
        return NULL;

    py_points = (PyArrayObject *)
        PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_occupancies = (PyArrayObject *)
        PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_points == NULL)
        goto fail;

    py_lmax = (PyArrayObject *)
        PyArray_FROM_OTF(arg3, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_lmax == NULL)
        goto fail;

    py_grid_to_cartesian = (PyArrayObject *)
        PyArray_FROM_OTF(arg5, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    py_out = (PyArrayObject *)
        PyArray_FROM_OTF(arg7, NPY_FLOAT64, NPY_ARRAY_INOUT_ARRAY);
    if (py_out == NULL)
        goto fail;

    // Get pointers to arrays and shape info.
    double *points = (double *) PyArray_DATA(py_points);
    double *occupancies = (double *) PyArray_DATA(py_occupancies);
    double *lmax = (double *) PyArray_DATA(py_lmax);
    double *grid_to_cartesian = (double *) PyArray_DATA(py_grid_to_cartesian);
    double *out = (double *) PyArray_DATA(py_out);

    npy_intp *points_shape = PyArray_DIMS(py_points);
    npy_intp *out_shape = PyArray_DIMS(py_out);

    // After the parsing and generation of pointers, the real function can
    // start.

    // Precompute some values.
    int out_slice = out_shape[2] * out_shape[1];
    int out_size = out_slice * out_shape[0];
    double rmax2 = SQUARE(rmax);

    int n;
    for (n = 0; n < points_shape[0]; n++) {
        double q = occupancies[n];
        if (q == 0) {
            continue;
        }

        int points_ind = 3 * n;
        double center_a = points[points_ind];
        double center_b = points[points_ind + 1];
        double center_c = points[points_ind + 2];

        int cmin = (int) ceil(center_c - lmax[2]);
        int bmin = (int) ceil(center_b - lmax[1]);
        int amin = (int) ceil(center_a - lmax[0]);

        int cmax = (int) floor(center_c + lmax[2]);
        int bmax = (int) floor(center_b + lmax[1]);
        int amax = (int) floor(center_a + lmax[0]);

        int c;
        for (c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * out_slice, out_size);
            double dc = center_c - c;
            double dist2_z = SQUARE(grid_to_cartesian[8] * dc);
            double disty_c = grid_to_cartesian[5] * dc;
            double distx_c = grid_to_cartesian[2] * dc;

            int b;
            for (b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * out_shape[2], out_slice) + ind_c;
                double db = center_b - b;
                double dist2_zy = dist2_z +
                    SQUARE(disty_c + grid_to_cartesian[4] * db);
                double distx_cb = distx_c + grid_to_cartesian[1] * db;

                int a;
                for (a = amin; a <= amax; a++) {
                    double da = center_a - a;
                    double dist2_zyx = dist2_zy +
                        SQUARE(distx_cb + grid_to_cartesian[0] * da);
                    if (dist2_zyx <= rmax2) {
                        out[ind_cb + modulo(a, out_shape[2])] = value;
                    }
                }
            }
        }
    }

    // Clean up objects
    Py_DECREF(py_points);
    Py_DECREF(py_occupancies);
    Py_DECREF(py_lmax);
    Py_DECREF(py_grid_to_cartesian);
    Py_DECREF(py_out);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    // Clean up objects
    Py_XDECREF(py_points);
    Py_XDECREF(py_occupancies);
    Py_XDECREF(py_lmax);
    Py_XDECREF(py_grid_to_cartesian);
    PyArray_DiscardWritebackIfCopy(py_out);
    Py_XDECREF(py_out);
    return NULL;
}


static PyMethodDef mymethods[] = {
    {"dilate_points", dilate_points, METH_VARARGS, ""},
    {"mask_points", mask_points, METH_VARARGS, ""},
    {"extend_to_p1", py_extend_to_p1, METH_VARARGS, ""},
    {"correlation_gradients", correlation_gradients, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
init_extensions(void)
{
    (void) Py_InitModule("_extensions", mymethods);
    import_array();
};

