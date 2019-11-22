__kernel void vector_add(__global const float *x,
                        __global const float *y,
                        __global float *restrict z)
{
  // uint current_dimension = get_work_dim();  // Gives us the dimensions of the kernel (we can create an array of kernel, a matrix of kernels etc...)
  size_t current_thread = get_global_id(0);  // Get the first id of the kernel (0 because we are in an array)
  z[current_thread] = x[current_thread] + y[current_thread];
}
