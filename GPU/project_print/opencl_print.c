#include <stdio.h>
#include <stdlib.h>
// #include <iostream> // for standard I/O
// #include <math.h>
// #include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024	// default length of the string buffer
// using namespace std;



void print_clbuild_errors(cl_program program,cl_device_id device) {
		printf("Program Build failed\n");
		size_t length;
		char buffer[STRING_BUFFER_LEN];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		printf("--- Build log ---\n %s\n",buffer);
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
  printf("file size %d\n",size);
  printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  printf("%s\n",*outputstr);
  printf("-------------------------------------------\n");
  return outputstr;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data) {
     fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
	if(status!=CL_SUCCESS)
		printf("%s\n",msg);
}


int main() {
	char char_buffer[STRING_BUFFER_LEN];
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_context_properties context_properties[] = {
		CL_CONTEXT_PLATFORM, 0,
		CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
		CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
		0
	};
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;


	char print_buffer[STRING_BUFFER_LEN];
	cl_mem print_buffer_cl; // num_devices elements
	int status;


	clGetPlatformIDs(1, &platform, NULL);
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
	printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
	printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
	printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

	context_properties[1] = (cl_context_properties)platform;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, device, 0, NULL);

	unsigned char **opencl_program=read_file("print.cl");
	program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
	if (program == NULL) {
		printf("Program creation failed\n");
		return 1;
	}

	int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
	kernel = clCreateKernel(program, "hello_world", NULL);

		// Input buffers.
	print_buffer_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, STRING_BUFFER_LEN * sizeof(char), NULL, &status);
	checkError(status, "Failed to create buffer for print");

	// Transfer inputs to each device. Each of the host buffers supplied to
	// clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
	// for the host-to-device transfer.
	cl_event kernel_event;

	// Set kernel arguments.
	unsigned argi = 0;

	status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &print_buffer_cl);
	checkError(status, "Failed to set argument print_buffer");

	const size_t global_work_size = 1;	// Only one thread for the print
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 2, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel");

	// Read the result. This the final operation.
	clWaitForEvents(1,&kernel_event);

	status = clEnqueueWriteBuffer(queue, print_buffer_cl, CL_TRUE,
			0, STRING_BUFFER_LEN * sizeof(char), print_buffer, 0, NULL, NULL);

	printf("%s\n", print_buffer);
	// Here CL_TmandQueue(queue);
	clReleaseMemObject(print_buffer_cl);
	clReleaseProgram(program);
	clReleaseContext(context);


	clFinish(queue);

	return 0;
}
