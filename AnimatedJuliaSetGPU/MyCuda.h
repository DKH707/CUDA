/// Defined in MyCuda.h, returns an error message for CUDA based operations. 
void myCudaErrorCheck(const char* file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line-1);
		exit(0);
	}
}
/// Defined in MyCuda.h, returns number of blocks (on x axis) needed based on number of items and threads/block.
int GridDimCalcX (const int n, const int blockSize)
{
	int blockCount = 0;
	
	blockCount=(n - 1)/blockSize + 1;

	return blockCount;
}
