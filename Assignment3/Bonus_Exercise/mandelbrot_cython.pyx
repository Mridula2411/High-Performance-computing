# cython: boundscheck=False
import numpy as np
cimport numpy as np




def mandelbrot(double complex c, int max_iter=100):

    cdef double complex z = 0
    cdef int n

    for n in range(max_iter):

        if (z.real * z.real + z.imag * z.imag) > 4:
            return n

        z = z * z + c
    return max_iter

def mandelbrot_set(int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter=100):

    cdef double [:, :] image = np.zeros((height, width), dtype=np.float64)

    cdef double[:] x_vals = np.linspace(x_min, x_max, width)
    cdef double[:] y_vals = np.linspace(y_min, y_max, height)

    cdef int i, j
    cdef double complex c

    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = mandelbrot(c, max_iter)

    return image