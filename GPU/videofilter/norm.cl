__kernel void norm(__global size_t * dimensions, __global unsigned char * input_a, __global unsigned char * input_b, __global unsigned char * output) {
  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t jmax = dimensions[1];
  uchar a = input_a[i * jmax + j];
  uchar b = input_b[i * jmax + j];
  output[i * jmax + j] = convert_uchar_sat_rte(sqrt((float) (a*a + b*b)));
}
