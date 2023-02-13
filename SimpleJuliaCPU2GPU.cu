//nvcc SimpleJuliaSetCPU2GPU.cu -o SimpleJuliaSetCPU2GPU -lglut -lGL -lm
// This is a simple Julia set which is repeated iterations of 
// Znew = Zold + C whre Z and Care imaginary numbers.
// After so many tries if Zinitial escapes color it black if it stays around color it red.


#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define A  -0.824  //real
#define B  -0.1711   //imaginary


unsigned int window_width = 1024;
unsigned int window_height = 1024;
float *GPUpixels, *CPUpixels;

dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.

void errorCheck(const char* file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

void AllocateMemory()
{
	CPUpixels = (float *)malloc(window_width*window_height*3*sizeof(float));
	cudaMalloc(&GPUpixels, window_width*window_height*3*sizeof(float));
}

void SetUpCudaDevices()
{
	BlockSize.x = window_width; //limited to 1024
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = window_height; 
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;


float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);


__device__ float color(float x, float y) 
{
	float mag,maxMag,temp;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		// Zn = Zo*Zo + C
		// or xn + yni = (xo + yoi)*(xo + yoi) + A + Bi
		// xn = xo*xo - yo*yo + A (real Part) and yn = 2*xo*yo + B (imaginary part)
		temp = x; // We will be changing the x but we need its old value to find y.	
		x = x*x - y*y + A;
		y = (2.0 * temp * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}


__global__ void Initialize(float xMin,float yMin,float stepSizeX,float stepSizeY, float* p)
{

	float x, y;
	int k;
	k=(threadIdx.x + blockDim.x * blockIdx.x) * 3;
	x = xMin + threadIdx.x * stepSizeX;
	y = yMin + blockIdx.x * stepSizeY;
	
		p[k] = color(x,y);	//Red on or off returned from color
		p[k+1] = 0.0; 		//Green off
		p[k+2] = 0.0;		//Blue off
		
}

void display(void) 
{ 
	Initialize<<<GridSize, BlockSize>>>(xMin, yMin, stepSizeX, stepSizeY, GPUpixels);
	errorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(CPUpixels, GPUpixels, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, CPUpixels); 
	glFlush(); 
}


int main(int argc, char** argv)
{ 
	SetUpCudaDevices();
	AllocateMemory();
	cudaMemcpyAsync(GPUpixels, CPUpixels, window_width*window_height*3*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
