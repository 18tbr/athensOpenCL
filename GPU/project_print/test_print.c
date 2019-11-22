#include <stdio.h>

void hello_world() {
  char buffer[1024];
  char *print_buffer = buffer;
  *print_buffer = "H";
  *(print_buffer+1) = "e";
  *(print_buffer+1) = "l";
  *(print_buffer+1) = "l";
  print_buffer[4] = "o";
  print_buffer[5] = " ";
  print_buffer[6] = "W";
  print_buffer[7] = "o";
  print_buffer[8] = "r";
  print_buffer[9] = "l";
  print_buffer[10] = "d";
  print_buffer[11] = "!";
  print_buffer[12] = "\n";
  printf("%s\n", buffer);
}
