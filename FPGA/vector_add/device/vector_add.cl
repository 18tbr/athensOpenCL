 // ACL kernel for adding two input vectors

__attribute__((num_compute_units(32)))
__attribute__((num_simd_work_items(4)))
__attribute__((reqd_work_group_size(256,1,1)))
// __kernel void vector_add(__global const float *restrict x,
//                         __global const float *restrict y,
//                         __global float *restrict z)
// {
//     // get index of the work item
//     int index = get_global_id(0);
//
//     // add the vector elements
//     z[index] = x[index] + y[index];
// }

__kernel void matrix_multiply(__global const float *restrict x,
                        __global const float *restrict y,
                        __global float *restrict z)
{
    // get index of the work item
    size_t index = get_global_id(0);
    size_t Nsquare = get_global_size(0); // Assuming sqare matrix of size N*N
    size_t N = convert_int_sat_rte(sqrt(Nsquare));
    // add the vector elements
    size_t i = (size_t) index / N;
    size_t j = index % N;

    z[index] = 0;
    for(size_t k = 0; k < N; k++)
    {
      z[index] += x[i*N+k]*y[k*N+j];
    }
}
