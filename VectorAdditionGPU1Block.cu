//Derek Hopkins
//Vector addition on the GPU with 1 block
//nvcc VectorAdditionGPU1Block.cu -o VectorAdditionGPU1Block


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>

//#include <sys/time.h> //LINUX

//sys/time.h equivalent for Windows system only *******************************
#define WIN32_LEAN_AND_MEAN 
#include <time.h>			
#include <winsock2.h>		
#include <Windows.h>		
#include <stdint.h>			
int gettimeofday(struct timeval* tp, struct timezone* tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
//*****************************************************

//Length of vectors to be added.
#define N 30

//Globals
float* A_CPU, * B_CPU, * C_CPU; //CPU pointers
float* A_GPU, * B_GPU, * C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float FloatNSize = N * sizeof(float);
//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = 200;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU, FloatNSize);
	cudaMalloc(&B_GPU, FloatNSize);
	cudaMalloc(&C_GPU, FloatNSize);

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(FloatNSize);
	B_CPU = (float*)malloc(FloatNSize);
	C_CPU = (float*)malloc(FloatNSize);
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;

	for (i = 0; i < N; i++)
	{
		A_CPU[i] = (float)2 * i;
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void AdditionGPU(float* a, float* b, float* c, int n)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	c[idx] = a[idx] + b[idx];
}

int main()
{
	int i;
	timeval start, end;

	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();

	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	//? ? ?
	cudaMemcpy(A_GPU, A_CPU, FloatNSize, cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, FloatNSize, cudaMemcpyHostToDevice);
	cudaMemcpy(C_GPU, C_CPU, FloatNSize, cudaMemcpyHostToDevice);

	//Calling the Kernel (GPU) function.	
	AdditionGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);

	//Copy Memory from GPU to CPU	
	//? ? ?

	cudaMemcpy(C_CPU, C_GPU, FloatNSize, cudaMemcpyDeviceToHost);
	
	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);

	// Displaying the vector. You will want to comment this out when the vector gets big.
	// This is just to make sure everything is running correctly.	
	for (i = 0; i < N; i++)
	{
		printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N - 1, A_CPU[N - 1], N - 1, B_CPU[N - 1], N - 1, C_CPU[N - 1]);

	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time / 1000.0));

	//You're done so cleanup your mess.
	CleanUp();

	return(0);
}
