__kernel void k_main(__global const float * mask, __global const char * matrix, __global char * output) {
  int i = (int) get_global_id(0);
  int j = (int) get_global_id(1);
  int imax = (int) get_global_size(0);
  int jmax = (int) get_global_size(1);
  // printf("%d",i);
  // printf("%d",j);
  // printf("GPU imax : %d ~ ",imax);
  // if (jmax!=640) printf("GPU jmax : %d ~ ",jmax);

  output[i*jmax+j] = 0;
  // printf("%d\n",imax*jmax);
  size_t count = 0;
  float buffer = 0;
  for (int ip = -1; ip < 2; ip++) {
    for (int jp = -1; jp < 2; jp++) {
      // printf("%d",( i + ip < imax ) && ( j + jp < jmax ) && ( i + ip >= 0 ) && ( j + jp >= 0 ));
      if ( (i+ip<imax) && (j+jp<jmax) && (i+ip+1>0) && (j+jp+1>0) ) {
        // printf("%d - ",matrix[(i+ip) * jmax + (j+jp)]);
        buffer += matrix[(i+ip)*jmax+(j+jp)] * mask[ip*3+jp];
      }
      // else add 0
    }
  }

  // printf("%d",count);
  // printf("%f - ",buffer);
  // printf("%d - ",convert_uchar_sat_rte(buffer));
  output[i*jmax+j] = convert_char_sat_rte(buffer);

  // printf("%d - ",convert_uchar_sat_rte(matrix[i*jmax+j] * mask[1*3+1]));
  // output[i*jmax+j] = matrix[i*jmax+j];
}
