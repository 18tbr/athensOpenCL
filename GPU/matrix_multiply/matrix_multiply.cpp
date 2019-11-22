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

int multiply(float *A, float *B, float *C, size_t dim);


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
    if(status!=CL_SUCCESS)
        printf("%s: %s\n",msg,getErrorString(status));
}

// Randomly generate a floating-point number between -1 and 1.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 2.0f - 1.0f;
}

int main(int argc, char *argv[])
{

		 clock_t global_start, global_stop;
		 unsigned Ninput = 256;
		 unsigned workgroupinput = 4;
		 printf("\nUsage : <executable> <size of matrix, default %d> <mode : copy or map, default map> <workgroup size, default %d>\n\n", Ninput, workgroupinput);

		 if (argc >= 2) {
 			Ninput = atoi(argv[1]);
 		}
 		const unsigned N = Ninput;

		printf("Square Matrix Size : %d x %d\n", N, N);

		 global_start = clock();

		 int map;
		 if ( (argc>=3 ) && ( strcmp(argv[2],"copy")==0 ) ) {
			printf("Mode : %s\n", "copy");
		 	map = 0;
		 }
		 else {
			 printf("Mode : %s\n", "map");
			 map = 1;
		 }

		 if (argc >= 4) {
		 	workgroupinput = atoi(argv[3]);
		 }

		 const unsigned workgroup = workgroupinput;
		 printf("Workgroup size : %d\n", workgroup);

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
     cl_command_queue queue;
     cl_program program;
     cl_kernel kernel;



//--------------------------------------------------------------------


		float *input_a = (float *) malloc(sizeof(float)*N*N);
		float *input_b = (float *) malloc(sizeof(float)*N*N);
		float *output = (float *) calloc(N*N,sizeof(float));
		float *ref_output = (float *) calloc(N*N,sizeof(float));
		cl_mem input_a_buf; // num_devices elements
		cl_mem input_b_buf; // num_devices elements2
		cl_mem output_buf; // num_devices elements
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

	   unsigned char **opencl_program=read_file("matrix_multiply.cl");
	   program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
	   if (program == NULL) {
       printf("Program creation failed\n");
       return 1;
		 }

		start = clock();

    int success=clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	  if(success!=CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "matrix_multiply", NULL);

		end = clock();
    diff = end - start;
    printf ("OpenCL took %.2ld ms to compile the code and create the kernel.\n", diff );
 		// Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
       N*N*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        N*N*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        N*N*sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");



    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
		cl_event kernel_event,finish_event;

		if (map  == 1) {
			start = clock();
			input_a = (float *)clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, CL_MAP_WRITE,
	        0, N*N*sizeof(float), 0, NULL, &write_event[0], &status);
	    checkError(status, "Failed to initialize input A");
	    input_b = (float *)clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, CL_MAP_WRITE,
	        0, N*N*sizeof(float), 0, NULL, &write_event[1], &status);
	    checkError(status, "Failed to initialize input B");
			end = clock();
			diff = end - start;

			printf("Time to get the input : %ld\n", diff);

			printf("%s\n", "Generating random matrix");

			start = clock();
			for(unsigned j = 0; j < N*N; ++j) {
			      input_a[j] = rand_float();
			      input_b[j] = rand_float();
			      //printf("ref %f\n",ref_output[j]);
	    }
			end = clock();
			diff = end - start;
			printf("Random matrix generation took %.2ld ms to run.\n", diff);

			cl_event unmap1, unmap2;
			// Required to prevent a bug, cf clEnqueueUnmapMemObject documentation
			clEnqueueUnmapMemObject(queue, input_a_buf, input_a, 0, NULL, &unmap1);
			clEnqueueUnmapMemObject(queue, input_b_buf, input_b, 0, NULL, &unmap2);

			clWaitForEvents(1,&unmap1);
			clWaitForEvents(1,&unmap2);

			printf("%s\n", "Running matrix multiplication on CPU");
			start = clock();
			multiply(input_a,input_b,ref_output,N);
			end = clock();
			diff = end - start;
	  	printf ("CPU took : %.2ld\n", diff);

		}
		else {
			printf("%s\n", "Generating random matrix");

			start = clock();
			for(unsigned j = 0; j < N*N; ++j) {
			      input_a[j] = rand_float();
			      input_b[j] = rand_float();
			      //printf("ref %f\n",ref_output[j]);
	    }
			end = clock();
			diff = end - start;
			printf("Random matrix generation took %.2ld ms to run.\n", diff);

			printf("%s\n", "Running matrix multiplication on CPU");
			start = clock();
			multiply(input_a,input_b,ref_output,N);
			end = clock();
			diff = end - start;
	  	printf ("CPU took : %2ld\n", diff);

			start = clock();
			status = clEnqueueWriteBuffer(queue, input_a_buf, CL_TRUE,
	        0, N*N*sizeof(float), input_a, 0, NULL, &write_event[0]);
	    checkError(status, "Failed to transfer input A");
	    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_TRUE,
	        0, N*N*sizeof(float), input_b, 0, NULL, &write_event[1]);
	    checkError(status, "Failed to transfer input B");
			end = clock();
			diff = end - start;
			printf("Time to get the vectors : %ld\n", diff);
		}

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

		printf("%s\n", "Running matrix multiplication on GPU.");

		start = clock();
    const size_t global_work_size[2] = {N,N};
		const size_t work_group_size[2] = {workgroup,workgroup};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_work_size, work_group_size, 2, write_event, &kernel_event);
		// status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
    //     global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");
    // Read the result. This the final operation.

		clWaitForEvents(1,&kernel_event);
		end = clock();
    diff = end - start;
		// printf("%ld\n", start);
		// printf("%ld\n", end);
		printf ("GPU took : %.2ld\n", diff);

		start = clock();
		if (map == 1) {
			output = (float *)clEnqueueMapBuffer(queue, output_buf, CL_TRUE, CL_MAP_READ,
	        0, N*N*sizeof(float), 1, &kernel_event, &finish_event, &status);
		}
		else {
			status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
	        0, N*N*sizeof(float), output, 1, &kernel_event, &finish_event);
		}
		end = clock();
		diff = end - start;
		printf("Reading time : %ld\n", diff);


/* code */
// Verify results.
bool pass = true;


// for (size_t i = 0; i < N*N; i++) {
// 	printf("%d\n",output[i]);
// }

for(unsigned j = 0; j < N*N && pass; ++j) {
      if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
        printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n",
            j, output[j], ref_output[j]);
        pass = false;
      }
}

	if (pass) {
		printf("%s\n", "GPU computation succeeded.");
	}

	start = clock();
  // Release local events.
  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(input_a_buf);
	clReleaseMemObject(input_b_buf);
	clReleaseMemObject(output_buf);
	clReleaseProgram(program);
	clReleaseContext(context);
	end = clock();
	diff = end - start;
	printf("Freeing time : %ld\n", diff);
	global_stop = clock();
	diff = global_stop - global_start;
	printf("Total execution time : %ld cycles\n", diff);


//--------------------------------------------------------------------

   clFinish(queue);
   return 0;
}





int multiply(float *A, float *B, float *C, size_t dim) {
	for (size_t i = 0; i < dim; i++) {
		for (size_t j = 0; j < dim; j++) {
			for (size_t k = 0; k < dim; k++) {
				C[i * dim + j] += A[i * dim + k] + B[k * dim + j];
			}
		}
	}
	return 0;
}
