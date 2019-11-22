#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define STRING_BUFFER_LEN 1024

using namespace cv;
using namespace std;





// ##################
// Easy cl struct for code reusability

struct easy_cl {
  cl_context context;
  cl_program program;
  cl_kernel kernel;
  cl_command_queue queue;
  int error;
};



struct convolution_buffers {
  cl_mem input_matrix_buf;
  cl_mem input_mask_buf;
  cl_mem output_matrix_buf;
};



struct norm_buffers {
  cl_mem input_matrixA_buf;
  cl_mem input_matrixB_buf;
  cl_mem output_matrix_buf;
};




// #################
// Function prototypes

const char *getErrorString(cl_int error);
void checkError(int status, const char *msg);
void print_clbuild_errors(cl_program program,cl_device_id device);
void callback(const char *buffer, size_t length, size_t final, void *user_data);
unsigned char ** read_file(const char *name);
int convolution(easy_cl ecl, convolution_buffers buffers, const size_t imax, const size_t jmax, const float *mask, const char *matrix, char *output);
int norm(easy_cl ecl, norm_buffers buffers, const size_t imax, const size_t jmax, const char *matrixA, const char *matrixB, char *output);
easy_cl compile(const char *fileName);
convolution_buffers set_convolution_buffers(easy_cl ecl, size_t imax, size_t jmax);
norm_buffers set_norm_buffers(easy_cl ecl, size_t imax, size_t jmax);







// ###################
// Main function

int main(int argc, char const *argv[]) {

  // ############################
  // Defining some global values
  float gaussian_blur_mask[9] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
	float gx_mask[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
	float gy_mask[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};

  // const char *convolutionCL = "convolution.cl";

  easy_cl convolution_cl = compile("convolution.cl");
  // printf("Convolution struct status %d\n", convolution_cl.error);

  easy_cl norm_cl = compile("norm.cl");


  // ############################
  // Loading video
  VideoCapture camera("./bourne.mp4");
  if(!camera.isOpened())  // check if we succeeded
      return -1;


  // #########################
  // Creating output
  const string NAME = "./output.avi";   // Form the new name with container
  int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
  Size S = Size((int) camera.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                (int) camera.get(CV_CAP_PROP_FRAME_HEIGHT));
  //Size S =Size(1280,720);
  cout << "SIZE:" << S << endl;
  VideoWriter outputVideo;                                        // Open the output
  outputVideo.open(NAME, ex, 25, S, true);
  if (!outputVideo.isOpened())
  {
      cout  << "Could not open the output video for write: " << NAME << endl;
      return -1;
  }

  size_t imax = S.height;
  size_t jmax = S.width;


  // Setting buffers before entering the loop.
  convolution_buffers conv_buffer = set_convolution_buffers(convolution_cl,imax,jmax);
  norm_buffers norm_buffer = set_norm_buffers(norm_cl,imax,jmax);


  time_t start,end;
	double diff;

  time(&start);

  // #########################
  // Filtering each frame.
  for (size_t frameCount = 0; frameCount < 299; frameCount++) {
    Mat cameraFrame;
    camera >> cameraFrame;  // Getting frame from the video

    Mat grayFrame;
    cvtColor(cameraFrame, grayFrame, CV_BGR2GRAY);  // Getting grayscale of the frame.

    // Getting data array from grayscale
    char array_a[imax*jmax];
    char array_b[imax*jmax];

    // Converting from uchar to char
    for (size_t k = 0; k < imax*jmax; k++) {
      array_a[k] = (char) grayFrame.data[k];
    }


    // ########################
    // Performing the convolutions for the gaussian blur
    convolution(convolution_cl,conv_buffer,imax,jmax,gaussian_blur_mask,array_a,array_b);

    convolution(convolution_cl,conv_buffer,imax,jmax,gaussian_blur_mask,array_b,array_a);

    convolution(convolution_cl,conv_buffer,imax,jmax,gaussian_blur_mask,array_a,array_b);
    // memcpy(input_array,output_array,imax*jmax*sizeof(char));  // piping


    // #######################
    // Performaing Sobel edge detection
    // Step 1 : convolution
    char gx_array[imax*jmax];
    char gy_array[imax*jmax];

    convolution(convolution_cl,conv_buffer,imax,jmax,gx_mask,array_b,gx_array);
    convolution(convolution_cl,conv_buffer,imax,jmax,gy_mask,array_b,gy_array);

    // Step 2 : summed norm
    norm(norm_cl,norm_buffer,imax,jmax,gx_array,gy_array,array_a);


    // Converting back from char to uchar
    unsigned char displayArray[imax*jmax];
    for (size_t k = 0; k < imax*jmax; k++) {
      displayArray[k] = array_a[k];
    }


    // Separating with a threshold.
    Mat edgeFrame = Mat(imax,jmax,CV_8U,displayArray);
    threshold(edgeFrame, edgeFrame, 80, 255, THRESH_BINARY);

    // Converting back to RGB for file format
    Mat displayFrame;
    grayFrame.copyTo(displayFrame,edgeFrame);
    cvtColor(displayFrame,displayFrame,CV_GRAY2BGR);

    // Registering the frame.
    outputVideo << displayFrame;
  }

  time(&end);
  diff = difftime(end,start);
  printf("FPS : %2lf\n", 299.0/diff);
  // Releasing ressources
  outputVideo.release();
	camera.release();

  return 0;
}








// #################################
// Auxiliary functions

const char *getErrorString(cl_int error) {
  switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILE2D";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTI2ES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVAcl_memLID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}


void checkError(int status, const char *msg) {
  if(status!=CL_SUCCESS) {
    printf("%s: %s\n",msg,getErrorString(status));
  }
}


void print_clbuild_errors(cl_program program,cl_device_id device) {
	cerr << "Program Build failed\n";
	size_t length;
	char buffer[2048];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
	cerr << "--- Build log ---\n "<<buffer<<endl;
	exit(1);
}

unsigned char ** read_file(const char *name) {
  size_t size;
  unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
  FILE* fp = fopen(name, "rb");
  if (!fp) {
    printf("no such file:%s",name);
    exit(-1);
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (unsigned char *)malloc(size);
  unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
  *outputstr= (unsigned char *)malloc(size);

  if (!*output) {
    fclose(fp);
    printf("mem allocate failure:%s",name);
    exit(-1);
  }

  if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
  fclose(fp);
  snprintf((char *)*outputstr,size,"%s\n",*output);
  return outputstr;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data) {
 fwrite(buffer, 1, length, stdout);
}




int convolution(easy_cl ecl, convolution_buffers buffers, const size_t imax, const size_t jmax, const float *mask, const char *matrix, char *output) {

  int status;

  // #######################
  // Copying input buffers
  cl_event write_event[2];
  status = clEnqueueWriteBuffer(ecl.queue, buffers.input_matrix_buf, CL_TRUE,
      0, imax*jmax*sizeof(char), matrix, 0, NULL, &write_event[0]);
  checkError(status, "Failed to transfer input matrix");

  status = clEnqueueWriteBuffer(ecl.queue, buffers.input_mask_buf, CL_TRUE,
      0, 3*3*sizeof(float), mask, 0, NULL, &write_event[1]);
  checkError(status, "Failed to transfer input mask");


  // ####################
  // Starting the kernel
  cl_event kernel_event;
  const size_t global_work_size[2] = {imax,jmax};
  const size_t work_group_size[2] = {4,4};
  status = clEnqueueNDRangeKernel(ecl.queue, ecl.kernel, 2, NULL,
      global_work_size, work_group_size, 2, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");


  // ####################
  // Reading the results
  cl_event finish_event;
  status = clEnqueueReadBuffer(ecl.queue, buffers.output_matrix_buf, CL_TRUE,
      0, imax*jmax*sizeof(char), output, 1, &kernel_event, &finish_event);
  checkError(status, "Failed to read output from kernel");
  clWaitForEvents(1,&kernel_event);


  return 0;
}





int norm(easy_cl ecl, norm_buffers buffers, const size_t imax, const size_t jmax, const char *matrixA, const char *matrixB, char *output) {

  int status;

  // #######################
  // Copying input buffers
  cl_event write_event[2];
  status = clEnqueueWriteBuffer(ecl.queue, buffers.input_matrixA_buf, CL_TRUE,
      0, imax*jmax*sizeof(char), matrixA, 0, NULL, &write_event[0]);
  checkError(status, "Failed to transfer input matrix A");

  status = clEnqueueWriteBuffer(ecl.queue, buffers.input_matrixB_buf, CL_TRUE,
      0, imax*jmax*sizeof(char), matrixB, 0, NULL, &write_event[1]);
  checkError(status, "Failed to transfer input matrix B");



  // ####################
  // Starting the kernel
  cl_event kernel_event;
  const size_t global_work_size[2] = {imax,jmax};
  const size_t work_group_size[2] = {4,4};
  status = clEnqueueNDRangeKernel(ecl.queue, ecl.kernel, 2, NULL,
      global_work_size, work_group_size, 2, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");


  // ####################
  // Reading the results
  cl_event finish_event;
  status = clEnqueueReadBuffer(ecl.queue, buffers.output_matrix_buf, CL_TRUE,
      0, imax*jmax*sizeof(char), output, 1, &kernel_event, &finish_event);
  checkError(status, "Failed to read output from kernel");
  clWaitForEvents(1,&kernel_event);


  return 0;
}





convolution_buffers set_convolution_buffers(easy_cl ecl, size_t imax, size_t jmax) {

  convolution_buffers result;
  int status;

  // #########################
  // Creating GPU buffers
  result.input_matrix_buf = clCreateBuffer(ecl.context, CL_MEM_READ_ONLY,
     imax*jmax*sizeof(char), NULL, &status);
  checkError(status, "Failed to create buffer for input matrix");

  result.input_mask_buf = clCreateBuffer(ecl.context, CL_MEM_READ_ONLY,
     3*3*sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for input mask");

  result.output_matrix_buf = clCreateBuffer(ecl.context, CL_MEM_WRITE_ONLY,
     imax*jmax*sizeof(char), NULL, &status);
  checkError(status, "Failed to create buffer for output matrix");


  // #####################
  // Setting kernel arguments
  unsigned argi = 0;
  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.input_mask_buf);
  checkError(status, "Failed to set argument input mask");

  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.input_matrix_buf);
  checkError(status, "Failed to set argument input matrix");

  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.output_matrix_buf);
  checkError(status, "Failed to set argument output matrix");

  return result;
}



norm_buffers set_norm_buffers(easy_cl ecl, size_t imax, size_t jmax) {

  int status;
  norm_buffers result;

  // #########################
  // Creating GPU buffers
  result.input_matrixA_buf = clCreateBuffer(ecl.context, CL_MEM_READ_ONLY,
     imax*jmax*sizeof(char), NULL, &status);
  checkError(status, "Failed to create buffer for input matrix A");

  result.input_matrixB_buf = clCreateBuffer(ecl.context, CL_MEM_READ_ONLY,
     imax*jmax*sizeof(char), NULL, &status);
  checkError(status, "Failed to create buffer for input matrix B");

  result.output_matrix_buf = clCreateBuffer(ecl.context, CL_MEM_WRITE_ONLY,
     imax*jmax*sizeof(char), NULL, &status);
  checkError(status, "Failed to create buffer for output matrix");


  // #####################
  // Setting kernel arguments
  unsigned argi = 0;
  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.input_matrixA_buf);
  checkError(status, "Failed to set argument input matrix A");

  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.input_matrixB_buf);
  checkError(status, "Failed to set argument input matrix B");

  status = clSetKernelArg(ecl.kernel, argi++, sizeof(cl_mem), &result.output_matrix_buf);
  checkError(status, "Failed to set argument output matrix");

  return result;
}









easy_cl compile(const char *fileName) {

  char char_buffer[STRING_BUFFER_LEN];
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_context_properties context_properties[] =
  {
       CL_CONTEXT_PLATFORM, 0,
       CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
       CL_PRINTF_BUFFERSIZE_ARM, 0x10000,
       0
  };

  cl_program program;
  cl_command_queue queue;
  cl_kernel kernel;


  easy_cl result;

  clGetPlatformIDs(1, &platform, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);

  context_properties[1] = (cl_context_properties)platform;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
  queue = clCreateCommandQueue(context, device, 0, NULL);

  unsigned char **opencl_program=read_file(fileName);
  program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
  if (program == NULL) {
    printf("Program creation failed\n");
    result.error = 1;
    return result;
  }

  int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
  kernel = clCreateKernel(program, "k_main", NULL);

  // sending values back to pointers
  result.context = context;
  result.kernel = kernel;
  result.program = program;
  result.queue = queue;
  result.error = 0;

  return result;
}
