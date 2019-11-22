__kernel void matrix_multiply(__global const float *x,
                        __global const float *y,
                        __global float *restrict z)
{
  // uint current_dimension = get_work_dim();  // Gives us the dimensions of the kernel (we can create an array of kernel, a matrix of kernels etc...)
  size_t i = get_global_id(0);  // Get the first id of the kernel (0 because we are in an array)
  size_t j = get_global_id(1);
  size_t dim = get_global_size(0);
  // printf("Global id %d - %d - %d - %d - %d\n", get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(3), get_global_id(4));
  // printf("%d - %d\n",i,j);
  // printf("%d\n",get_work_dim());
  // printf("%d\n - %d\n - %d\n",get_global_size(0),get_global_size(1),get_global_size(2));
  // printf("%d\n",get_work_dim());
  for (size_t k = 0; k < dim; k++) {
    z[i * dim + j] += x[i * dim + k] + y[dim * k + j];
  }
}
