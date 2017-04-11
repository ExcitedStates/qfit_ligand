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

    for (int n = 0; n < points_shape[0]; n++) {
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

        for (int c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * out_slice, out_size);
            double dc = center_c - c;
            double dist2_z = SQUARE(grid_to_cartesian[8] * dc);
            double disty_c = grid_to_cartesian[5] * dc;
            double distx_c = grid_to_cartesian[2] * dc;

            for (int b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * out_shape[2], out_slice) + ind_c;
                double db = center_b - b;
                double dist2_zy = dist2_z + 
                    SQUARE(disty_c + grid_to_cartesian[4] * db);
                double distx_cb = distx_c + grid_to_cartesian[1] * db;

                for (int a = amin; a <= amax; a++) {
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
    PyArray_XDECREF_ERR(py_out);
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

    for (int n = 0; n < points_shape[0]; n++) {
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

        for (int c = cmin; c <= cmax; c++) {

            int ind_c = modulo(c * out_slice, out_size);
            double dc = center_c - c;
            double dist2_z = SQUARE(grid_to_cartesian[8] * dc);
            double disty_c = grid_to_cartesian[5] * dc;
            double distx_c = grid_to_cartesian[2] * dc;

            for (int b = bmin; b <= bmax; b++) {
                int ind_cb = modulo(b * out_shape[2], out_slice) + ind_c;
                double db = center_b - b;
                double dist2_zy = dist2_z + 
                    SQUARE(disty_c + grid_to_cartesian[4] * db);
                double distx_cb = distx_c + grid_to_cartesian[1] * db;

                for (int a = amin; a <= amax; a++) {
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
    PyArray_XDECREF_ERR(py_out);
    return NULL;
}


static PyMethodDef mymethods[] = {
    {"dilate_points", dilate_points, METH_VARARGS, ""},
    {"mask_points", mask_points, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
init_extensions(void)
{
    (void) Py_InitModule("_extensions", mymethods);
    import_array();
};

