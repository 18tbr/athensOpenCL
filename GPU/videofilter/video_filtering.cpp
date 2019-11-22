#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#define STRING_BUFFER_LEN 1024
using namespace std;

#include <fstream>
#include "opencv2/opencv.hpp"

using namespace cv;
// #define SHOW


void print_clbuild_errors(cl_program program,cl_device_id device)
	{
		cout<<"Program Build failed\n";
		size_t length;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		cout<<"--- Build log ---\n "<<buffer<<endl;
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
  // printf("file size %d\n",size);
  // printf("-------------------------------------------\n");
  snprintf((char *)*outputstr,size,"%s\n",*output);
  // printf("%s\n",*outputstr);
  // printf("-------------------------------------------\n");
  return outputstr;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


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
    case -8: return "CL_MEM_COPY_OVER";
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
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
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
    case -57: return "CL_INVALID_EVEN";
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
    if(status!=CL_SUCCESS)
        printf("%s: %s\n",msg,getErrorString(status));
}





int norm(const char *argv, const size_t *dimensions, const unsigned char *A, const unsigned char *B, unsigned char *output) {
		 clock_t global_start, global_stop;
		 global_start = clock();

		 int map;
		 if (strcmp(argv,"copy")==0) {
			printf("%s\n", "Copy mode");
		 	map = 0;
		 }
		 else {
			 printf("%s\n", "Map mode");
			 map = 1;
		 }

     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     {
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;

		 size_t imax = dimensions[0];
		 size_t jmax = dimensions[1];

//--------------------------------------------------------------------

		unsigned char *input_a = (unsigned char *) malloc(sizeof(char)*imax*jmax);
		unsigned char *input_b = (unsigned char *) malloc(sizeof(char)*imax*jmax);
		// unsigned char *output_matrix = (unsigned char *) calloc(imax*jmax,sizeof(char));
		size_t *dimensions_array = (size_t *) malloc(2 * sizeof(size_t));
		// float *ref_output = (float *) calloc(N*N,sizeof(float));
		cl_mem input_a_buf; // num_devices elements
		cl_mem input_b_buf; // num_devices elements2
		cl_mem output_buf; // num_devices elements
		cl_mem dimensions_buf;
		int status;

		clock_t start,end;
		long int diff;

    // time (&start);
	   clGetPlatformIDs(1, &platform, NULL);
	   clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	   clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	   clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

	   context_properties[1] = (cl_context_properties)platform;
	   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	   context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
	   queue = clCreateCommandQueue(context, device, 0, NULL);

	   unsigned char **opencl_program=read_file("norm.cl");
	   program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
	   if (program == NULL) {
       // printf("Program creation failed\n");
       return 1;
		 }

		start = clock();

    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "norm", NULL);

		end = clock();
    diff = end - start;
    // printf ("OpenCL took %.2ld ms to compile the code and create the kernel.\n", diff );
 		// Input buffers.
		dimensions_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       2*sizeof(size_t), NULL, &status);
    checkError(status, "Failed to create buffer for dimensions");

    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        imax*jmax*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for matrix");

		input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        imax*jmax*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for matrix");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        imax*jmax*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for output");



    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[3];
		cl_event kernel_event,finish_event;

		if (map  == 1) {
			start = clock();
	    input_a = (unsigned char *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, CL_MAP_WRITE,
	        0, imax*jmax*sizeof(char), 0, NULL, &write_event[0], &status);
			input_b = (unsigned char *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, CL_MAP_WRITE,
	        0, imax*jmax*sizeof(char), 0, NULL, &write_event[1], &status);
			dimensions_array = (size_t *)clEnqueueMapBuffer(queue, dimensions_buf, CL_TRUE, CL_MAP_WRITE,
	        0, 2*sizeof(size_t), 0, NULL, &write_event[2], &status);
	    checkError(status, "Failed to initialize input B");
			end = clock();
			diff = end - start;

			// printf("Time to get the input : %ld\n", diff);

			// printf("%s\n", "Copying input matrix to buffer");

			start = clock();
			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_a[j] = A[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_b[j] = B[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for (size_t i = 0; i < 2; i++) {
				dimensions_array[i] = dimensions[i];
			}

			end = clock();
			diff = end - start;
			// printf("Copying matrix took %.2ld ms to run.\n", diff);

			cl_event unmap1, unmap2, unmap3;
			// Required to prevent a bug, cf clEnqueueUnmapMemObject documentation
			clEnqueueUnmapMemObject(queue, input_a_buf, input_a, 0, NULL, &unmap1);
			clEnqueueUnmapMemObject(queue, input_b_buf, input_b, 0, NULL, &unmap2);
			clEnqueueUnmapMemObject(queue, dimensions_buf, dimensions_array, 0, NULL, &unmap3);

			clWaitForEvents(1,&unmap1);
			clWaitForEvents(1,&unmap2);
			clWaitForEvents(1,&unmap3);
		}
		else {
			// printf("%s\n", "Copying input matrix to buffer");

			start = clock();
			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_a[j] = A[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_b[j] = B[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for (size_t i = 0; i < 2; i++) {
				dimensions_array[i] = dimensions[i];
			}

			end = clock();
			diff = end - start;
			// printf("Copying matrix took %.2ld ms to run.\n", diff);

			start = clock();
	    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_TRUE,
	        0, imax*jmax*sizeof(char), input_a, 0, NULL, &write_event[0]);
			checkError(status, "Failed to transfer input B");
			status = clEnqueueWriteBuffer(queue, input_b_buf, CL_TRUE,
	        0, imax*jmax*sizeof(char), input_b, 0, NULL, &write_event[1]);
			checkError(status, "Failed to transfer input B");
			status = clEnqueueWriteBuffer(queue, dimensions_buf, CL_TRUE,
					0, 2*sizeof(float), dimensions_array, 0, NULL, &write_event[2]);
	    checkError(status, "Failed to transfer input B");
			end = clock();
			diff = end - start;
			// printf("Time to get the vectors : %ld\n", diff);
		}

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &dimensions_buf);
    checkError(status, "Failed to set argument 1");

		status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 4");

		// printf("%s\n", "Running matrix summed norm on GPU.");

		start = clock();
    const size_t global_work_size[2] = {imax,jmax};
		// const size_t work_group_size[2] = {4,4};
    // status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
    //     global_work_size, work_group_size, 2, write_event, &kernel_event);
		status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, NULL, 3, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.

		clWaitForEvents(1,&kernel_event);
		end = clock();
    diff = end - start;
		// printf("%ld\n", start);
		// printf("%ld\n", end);
		// printf ("GPU took %.2ld ms to run.\n", diff);

		start = clock();
		if (map == 1) {
			output = (unsigned char *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ,
	        0, imax*jmax*sizeof(char), 1, &kernel_event, &finish_event, &status);
		}
		else {
			status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
	        0, imax*jmax*sizeof(char), output, 1, &kernel_event, &finish_event);
		}
		end = clock();
		diff = end - start;
		// printf("Reading time : %ld\n", diff);


/* code */
// Verify results.
bool pass = true;


// for (size_t i = 0; i < N*N; i++) {
// 	printf("%d\n",output[i]);
// }

// for(unsigned j = 0; j < N*N && pass; ++j) {
//       if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
//         printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
//             j, output[j], ref_output[j]);
//         pass = false;
//       }
// }

	if (pass) {
		// printf("%s\n", "GPU computation succeeded.");
	}

	start = clock();
  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
	clReleaseEvent(write_event[2]);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_a_buf);
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(dimensions_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);
	end = clock();
	diff = end - start;
	// printf("Freeing time : %ld\n", diff);
	global_stop = clock();
	diff = global_stop - global_start;
	printf("Total execution time : %ld cycles\n", diff);


//--------------------------------------------------------------------

   clFinish(queue);
   return 0;
}










int convolution(const char *argv, const float *mask, const unsigned char *matrix, const size_t *dimensions, unsigned char *output) {
		 // clock_t global_start, global_stop;
		 // global_start = clock();
		 int map;
		 if (strcmp(argv,"copy")==0) {
			// printf("%s\n", "Copy mode");
		 	map = 0;
		 }
		 else {
			 // printf("%s\n", "Map mode");
			 map = 1;
		 }

     char char_buffer[STRING_BUFFER_LEN];
     cl_platform_id platform;
     cl_device_id device;
     cl_context context;
     cl_context_properties context_properties[] =
     {
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;

		 size_t imax = dimensions[0];
		 size_t jmax = dimensions[1];

//--------------------------------------------------------------------

		unsigned char *input_matrix = (unsigned char *) malloc(sizeof(unsigned char)*imax*jmax);
		float *input_mask = (float *) malloc(sizeof(float)*3*3);
		// unsigned char *output_matrix = (unsigned char *) calloc(imax*jmax,sizeof(char));
		size_t *dimensions_array = (size_t *) malloc(2 * sizeof(size_t));
		unsigned char *output_array = (unsigned char *) calloc(imax*jmax,sizeof(unsigned char));
		cl_mem input_mask_buf; // num_devices elements
		cl_mem input_matrix_buf; // num_devices elements2
		cl_mem output_buf; // num_devices elements
		cl_mem dimensions_buf;
		int status;

		// clock_t start,end;
		// long int diff;

    // time (&start);
	   clGetPlatformIDs(1, &platform, NULL);
	   clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
	   clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
	   clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
	   // printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

	   context_properties[1] = (cl_context_properties)platform;
	   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	   context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
	   queue = clCreateCommandQueue(context, device, 0, NULL);

	   unsigned char **opencl_program=read_file("convolution.cl");
	   program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
	   if (program == NULL) {
       // printf("Program creation failed\n");
       return 1;
		 }

		// start = clock();

    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "convolution", NULL);

		// end = clock();
    // diff = end - start;
    // printf ("OpenCL took %.2ld ms to compile the code and create the kernel.\n", diff );
 		// Input buffers.
    input_mask_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       3*3*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for mask");

		dimensions_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       2*sizeof(size_t), NULL, &status);
    checkError(status, "Failed to create buffer for dimensions");

    input_matrix_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        imax*jmax*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for matrix");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        imax*jmax*sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for output");


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[3];
		cl_event kernel_event;

		if (map  == 1) {
			// start = clock();
			input_mask = (float *)clEnqueueMapBuffer(queue, input_mask_buf, CL_TRUE, CL_MAP_WRITE,
	        0, 3*3*sizeof(float), 0, NULL, &write_event[0], &status);
	    checkError(status, "Failed to initialize input A");
	    input_matrix = (unsigned char *)clEnqueueMapBuffer(queue, input_matrix_buf, CL_TRUE, CL_MAP_WRITE,
	        0, imax*jmax*sizeof(char), 0, NULL, &write_event[1], &status);
			dimensions_array = (size_t *)clEnqueueMapBuffer(queue, dimensions_buf, CL_TRUE, CL_MAP_WRITE,
	        0, 2*sizeof(size_t), 0, NULL, &write_event[2], &status);
	    checkError(status, "Failed to initialize input B");
			// end = clock();
			// diff = end - start;

			// printf("Time to get the input : %ld\n", diff);

			// printf("%s\n", "Copying input matrix to buffer");

			// start = clock();
			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_matrix[j] = matrix[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for (size_t i = 0; i < 3*3; i++) {
				input_mask[i] = mask[i];
				/* code */
			}

			for (size_t i = 0; i < 2; i++) {
				dimensions_array[i] = dimensions[i];
			}

			// end = clock();
			// diff = end - start;
			// printf("Copying matrix took %.2ld ms to run.\n", diff);
			cl_event unmap1, unmap2, unmap3;
			// Required to prevent a bug, cf clEnqueueUnmapMemObject documentation
			clEnqueueUnmapMemObject(queue, input_mask_buf, input_mask, 0, NULL, &unmap1);
			clEnqueueUnmapMemObject(queue, input_matrix_buf, input_matrix, 0, NULL, &unmap2);
			clEnqueueUnmapMemObject(queue, dimensions_buf, dimensions_array, 0, NULL, &unmap3);

			clWaitForEvents(1,&unmap1);
			clWaitForEvents(1,&unmap2);
			clWaitForEvents(1,&unmap3);
		}
		else {
			// printf("%s\n", "Copying input matrix to buffer");

			// start = clock();
			for(size_t j = 0; j < imax*jmax; ++j) {
		      input_matrix[j] = matrix[j];
			      //printf("ref %f\n",ref_output[j]);
	    }

			for (size_t i = 0; i < 3*3; i++) {
				input_mask[i] = mask[i];
				/* code */
			}

			dimensions_array[0] = dimensions[0];
			dimensions_array[1] = dimensions[1];
			// for (size_t i = 0; i < 2; i++) {
			// 	dimensions_array[i] = dimensions[i];
			// }

			// end = clock();
			// diff = end - start;
			// printf("Copying matrix took %.2ld ms to run.\n", diff);
			// start = clock();
			status = clEnqueueWriteBuffer(queue, input_mask_buf, CL_TRUE,
	        0, 3*3*sizeof(float), input_mask, 0, NULL, &write_event[0]);
	    checkError(status, "Failed to transfer input A");
	    status = clEnqueueWriteBuffer(queue, input_matrix_buf, CL_TRUE,
	        0, imax*jmax*sizeof(char), input_matrix, 0, NULL, &write_event[1]);
			checkError(status, "Failed to transfer input B");
			status = clEnqueueWriteBuffer(queue, dimensions_buf, CL_TRUE,
					0, 2*sizeof(size_t), dimensions_array, 0, NULL, &write_event[2]);
	    checkError(status, "Failed to transfer input B");
			// end = clock();
			// diff = end - start;
			// printf("Time to get the vectors : %ld\n", diff);

			clWaitForEvents(3,write_event);
		}

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_mask_buf);
    checkError(status, "Failed to set argument 1");
		// memcpy(input_matrix,output_mat
		status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &dimensions_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_matrix_buf);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 4");
		// printf("%s\n", "Running matrix convolution on GPU.");

		// start = clock();
    const size_t global_work_size[2] = {imax,jmax};
		const size_t work_group_size[2] = {4,4};
    // status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
    //     global_work_size, work_group_size, 2, write_event, &kernel_event);
		status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, work_group_size, 3, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.

		clWaitForEvents(1,&kernel_event);
		// end = clock();
    // diff = end - start;
		// printf("%ld\n", start);
		// printf("%ld\n", end);
		// printf ("GPU took %.2ld ms to run.\n", diff);

		cl_event finish_event;
		// start = clock();
		if (map == 1) {
			// cl_event unmap_output;
			output_array = (unsigned char *)clEnqueueMapBuffer(queue, output_buf, CL_FALSE, CL_MAP_READ,
	        0, imax*jmax*sizeof(char), 1, &kernel_event, &finish_event, &status);
			// clEnqueueUnmapMemObject(queue, output_buf, output, 1, &finish_event, &unmap_output);
			// clWaitForEvents(1,&unmap_output);
			checkError(status,"Failed to read result");
		}
		else {
			status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
	        0, imax*jmax*sizeof(char), output_array, 1, &kernel_event, &finish_event);
			checkError(status,"Failed to read result");
		}
		clWaitForEvents(1,&finish_event);
		// end = clock();
		for (size_t i = 0; i < imax*jmax; i++) {
			output[i] = output_array[i];
		}
		// diff = end - start;
		// printf("Reading time : %ld\n", diff);


/* code */
// Verify results.
bool pass = true;

// // DEBUG STUFF
// printf("%s\n","");
// // DEBUG STUFF
// for(size_t j = 0; j < imax*jmax; ++j) {
// 		printf("%d", output[j]);
// 			//printf("ref %f\n",ref_output[j]);
// }
// // DEBUG STUFF
// printf("%s\n","");

// for (size_t i = 0; i < N*N; i++) {
// 	printf("%d\n",outconvolutionput[i]);
// }

// for(unsigned j = 0; j < N*N && pass; ++j) {
//       if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
//         printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
//             j, output[j], ref_output[j]);
//         pass = false;
//       }
// }

	if (pass) {
		// printf("%s\n", "GPU computation succeeded.");
	}

	// start = clock();// output[i * jmax + j] = 0;
  // float buffer = 0;
  // for (int ip = -1; ip < 2; ip++) {
  //   for (int jp = -1; jp < 2; jp++) {
  //     // printf("%d",( i + ip < imax ) && ( j + jp < jmax ) && ( i + ip >= 0 ) && ( j + jp >= 0 ));
  //     if ( ( i + ip < imax ) && ( j + jp < jmax ) && ( i + ip >= 0 ) && ( j + jp >= 0 ) ) {
  //       // printf("%d - ",matrix[(i+ip) * jmax + (j+jp)]);
  //       buffer += matrix[(i+ip)*jmax+(j+jp)] * mask[ip*3+jp];
  //     }
  //     // else add 0
  //   }
  // }
  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
	clReleaseEvent(write_event[2]);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_mask_buf);
	clReleaseMemObject(input_matrix_buf);
	clReleaseMemObject(dimensions_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);
	// end = clock();
	// diff = end - start;
	// printf("Freeing time : %ld\n", diff);
	// global_stop = clock();
	// diff = global_stop - global_start;
	// printf("Total execution time : %ld cycles\n", diff);


//--------------------------------------------------------------------

   clFinish(queue);
   return 0;
}






int main(int argc, char const *argv[]) {
	float gaussian_blur_mask[9] = {1/12,1/4,1/12,1/4,1/3,1/4,1/12,1/4,1/12};
	// float gx_mask[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	// float gy_mask[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

	char mode[5];
	if ( (argc == 2) && (strcmp(argv[1],"copy")==0) ) {
		sprintf(mode,"%s","copy");
	}
	else {
		sprintf(mode,"%s","map");
	}

	VideoCapture camera("./bourne.mp4");
	if(!camera.isOpened())  // check if we succeeded
			return -1;




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
	clock_t start,end;
	long int diff,tot;
	tot = 0;	// Initialisation
	int count=0;
	// const char *windowName = "filter";   // Name shown in the GUI window.
	// #ifdef SHOW
	// namedWindow(windowName); // Resizable window, mightedge not work on Windows.
	// #endif



	while (true) {

		// First step is to get the grayscale value of the image.
			Mat cameraFrame,displayframe;
			count=count+1;
			// printf("%d\n", count);
			if(count > 299) break;
			camera >> cameraFrame;
			size_t dimensions_array[2] = {(size_t) cameraFrame.size().height,(size_t) cameraFrame.size().width};
			Mat grayframe,edge_x,edge_y,edge,edge_inv;
		cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);


  start = clock();

		unsigned char *input_matrix = (unsigned char *) malloc(dimensions_array[0]*dimensions_array[1]*sizeof(char));
		unsigned char *output_matrix = (unsigned char *) calloc(dimensions_array[0]*dimensions_array[1],sizeof(char));

		memcpy(input_matrix,grayframe.data,dimensions_array[0]*dimensions_array[1]*sizeof(char));	// Copying image.
		// // DEBUG STUFF
		// memcpy(output_matrix,grayframe.data,dimensions_array[0]*dimensions_array[1]*sizeof(char));	// Copying image.

		convolution(mode,gaussian_blur_mask,input_matrix,dimensions_array,output_matrix);
		memcpy(input_matrix,output_matrix,dimensions_array[0]*dimensions_array[1]*sizeof(char));	// Saving result to compute several gaussian blurs.


		convolution(mode,gaussian_blur_mask,input_matrix,dimensions_array,output_matrix);
		memcpy(input_matrix,output_matrix,dimensions_array[0]*dimensions_array[1]*sizeof(char));	// Saving result to compute several gaussian blurs.

		convolution(mode,gaussian_blur_mask,input_matrix,dimensions_array,output_matrix);
		memcpy(input_matrix,output_matrix,dimensions_array[0]*dimensions_array[1]*sizeof(char));	// Saving result to compute several gaussian blurs.
	//
	// 	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
	// 	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
	// 	GaussianBlur(grayframe, grayframe, Size(3,3),0,0);
	// Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
	// Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
	// addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
	//
	// 	// Computing Sobel edge detection
		// unsigned char *gx_matrix = (unsigned char *) calloc(dimensions_array[0]*dimensions_array[1],sizeof(char));
		// unsigned char *gy_matrix = (unsigned char *) calloc(dimensions_array[0]*dimensions_array[1],sizeof(char));
		//
		// convolution(mode,gx_mask,input_matrix,dimensions_array,gx_matrix);	// Gx filter
		// convolution(mode,gy_mask,input_matrix,dimensions_array,gy_matrix);	// Gy filter
		//
		// norm(mode, dimensions_array, gx_matrix, gy_matrix, output_matrix);


			// threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
			// printf("%d - %d\n",dimensions_array[0],dimensions_array[1]);
			edge = Mat(dimensions_array[0],dimensions_array[1],CV_8U,output_matrix);

			// // DEBUG STUFF
			// edge = Mat(dimensions_array[0],dimensions_array[1],CV_8U,calloc(dimensions_array[0]*dimensions_array[1],sizeof(unsigned char)));
			//
			end = clock();
			cvtColor(edge, edge_inv, CV_GRAY2BGR);
			// Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
			// memset((char*)displayframe.data, 0, dimensions_array[0]*dimensions_array[1]);

			displayframe = Mat(dimensions_array[0], dimensions_array[1],CV_8U);
			memset((char*)displayframe.data, 0, displayframe.step * displayframe.rows);

			// // DEBUG STUFF
			// printf("Frame size : %d - %d\n", grayframe.size().width,grayframe.size().height);
			// printf("Target size : %d - %d\n", displayframe.size().width,displayframe.size().height);
			// printf("Mask size : %d - %d\n", edge.size().width,edge.size().height);

			// grayframe.copyTo(displayframe,edge_inv);
			// grayframe.copyTo(displayframe);
			edge.copyTo(displayframe);
			cvtColor(displayframe, displayframe, CV_GRAY2BGR);
	outputVideo << displayframe;

	// // DEBUG STUFF
	// cvtColor(edge, edge, CV_GRAY2BGR);
	// outputVideo << edge;
	// outputVideo << grayframe;
	#ifdef SHOW
			imshow(windowName, displayframe);
	#endif
	diff = end - start;
	//printf("diff %ld = end %ld - start %ld\n", diff, end, start);
	tot+=diff;
	}
	outputVideo.release();
	camera.release();
	printf ("FPCycle %.2lf.\n", 299.0/tot );

	return EXIT_SUCCESS;
}
