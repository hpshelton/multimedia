#define CLAMP(a) ((a>255) ? 255 : ((a<0) ? 0 : (unsigned char)(a)))

__global__ void setToVal(unsigned char* x, int len, int val)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < len)
		x[index] = val;	
}

__global__ void conv3x3(unsigned char* input, unsigned char* output, int row, int col, float* kernel)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (xIndex + yIndex * row*4);

	if(index < row*col*4){
		int i, j;
		float convSum=0;
		for(i=-1; i < 2; i++){
			for(j=-1; j < 2; j++){
				if(-1 < (index+4*j)+(4*col*i) && (index+4*j)+(4*col*i) < row*col*4){
					convSum += kernel[3*(i+1) + (j+1)]*input[(index+4*j)+(4*col*i)];
				}
			}
		}
		output[index] = CLAMP(convSum);
	}
}

__global__ void quantize(float* x, int Qlevel, float maxval, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<n)
		x[i] = (int)( (x[i]/maxval) *Qlevel ) * (maxval / (float)Qlevel);
}

__global__ void brighten(unsigned char* input, unsigned char* output, int row, int col, float factor)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (xIndex + yIndex * row*4);

	if(index < row*col*4){
		output[index]=CLAMP(factor*input[index]);
	}
}

__global__ void greyscale(unsigned char* input, unsigned char* output, int row, int col)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (4*xIndex + yIndex * row*4);

	if(index < row*col*4){
		int lum = 0.11*input[index] + 0.59*input[index+1] + 0.3*input[index+2];

		output[index]=lum;
		output[index+1]=lum;
		output[index+2]=lum;
		output[index+3]=0;
	}
}

__global__ void saturate(unsigned char* input, unsigned char* output, int row, int col, float factor)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (4*xIndex + yIndex * row*4);

	if(index < row*col*4){
		int lum = 0.11*input[index] + 0.59*input[index+1] + 0.3*input[index+2];

		output[index]=CLAMP( (1-factor)*lum + factor*input[index] );
		output[index+1]=CLAMP( (1-factor)*lum + factor*input[index+1] );
		output[index+2]=CLAMP( (1-factor)*lum + factor*input[index+2] );
		output[index+3]=0;
	}
}

__global__ void fwt97(float* output, unsigned char* input, float* tempbank, int n)
{
	float a;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<n){
		output[i] = input[i];
		__syncthreads();

		// Predict 1
		a = -1.586134342;
		if(i%2==1 && i!=(n-1)){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==(n-1))
			output[n-1]+=2*a*output[n-2];
		__syncthreads();

		// Update 1
		a = -0.05298011854;
		if(i%2==0 && i!=0){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==0)
			output[0]+=2*a*output[1];
		__syncthreads();

		// Predict 2
		a = 0.8829110762;
		if(i%2==1 && i!=(n-1)){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==(n-1))
			output[n-1]+=2*a*output[n-2];
		__syncthreads();

		// Update 2
		a = 0.4435068522;
		if(i%2==0 && i!=0){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==0)
			output[0]+=2*a*output[1];
		__syncthreads();

		// Scale
		a = 1/1.149604398;
			if (i%2)
				output[i]*=a;
			else
				output[i]/=a;
		__syncthreads();

		// Pack
			if (i%2==0)
				tempbank[i/2]=output[i];
			else
				tempbank[n/2+i/2]=output[i];
		__syncthreads();

		output[i]=tempbank[i];
	}
}

__global__ void fwt97(float* output, float* tempbank, int n)
{
	float a;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<n){

		// Predict 1
		a = -1.586134342;
		if(i%2==1 && i!=(n-1)){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==(n-1))
			output[n-1]+=2*a*output[n-2];
		__syncthreads();

		// Update 1
		a = -0.05298011854;
		if(i%2==0 && i!=0){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==0)
			output[0]+=2*a*output[1];
		__syncthreads();

		// Predict 2
		a = 0.8829110762;
		if(i%2==1 && i!=(n-1)){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==(n-1))
			output[n-1]+=2*a*output[n-2];
		__syncthreads();

		// Update 2
		a = 0.4435068522;
		if(i%2==0 && i!=0){
			output[i]+=a*(output[i-1]+output[i+1]);
		}
		if(i==0)
			output[0]+=2*a*output[1];
		__syncthreads();

		// Scale
		a = 1/1.149604398;
			if (i%2)
				output[i]*=a;
			else
				output[i]/=a;
		__syncthreads();

		// Pack
			if (i%2==0)
				tempbank[i/2]=output[i];
			else
				tempbank[n/2+i/2]=output[i];
		__syncthreads();

		output[i]=tempbank[i];
	}
}

__global__ void iwt97(unsigned char* output, float* input, float* tempbank, int n)
{
	float a;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<n){
		// Unpack
		if(i<(n/2)){
			tempbank[i*2]=input[i];
			tempbank[i*2+1]=input[i+n/2];
		}
		__syncthreads();
		input[i]=tempbank[i];
		__syncthreads();

		// Undo scale
		a=1.149604398;
			if (i%2)
				input[i]*=a;
			else
				input[i]/=a;
		__syncthreads();

		// Undo update 2
		a=-0.4435068522;
		if(i%2==0 && i!=0){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==0)
			input[0]+=2*a*input[1];
		__syncthreads();

		// Undo predict 2
		a=-0.8829110762;
		if(i%2==1 && i!=(n-1)){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==(n-1))
			input[n-1]+=2*a*input[n-2];
		__syncthreads();

		// Undo update 1
		a=0.05298011854;
		if(i%2==0 && i!=0){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==0)
			input[0]+=2*a*input[1];
		__syncthreads();

		// Undo predict 1
		a=1.586134342;
		if(i%2==1 && i!=(n-1)){
			input[i]+=a*(input[i-1]+input[i+1]);
		} 
		if(i==(n-1))
			input[n-1]+=2*a*input[n-2];
		__syncthreads();

		output[i]=CLAMP(input[i]);
	}
}

__global__ void iwt97(float* input, float* tempbank, int n)
{
	float a;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<n){
		// Unpack
		if(i<(n/2)){
			tempbank[i*2]=input[i];
			tempbank[i*2+1]=input[i+n/2];
		}
		__syncthreads();
		input[i]=tempbank[i];
		__syncthreads();

		// Undo scale
		a=1.149604398;
			if (i%2)
				input[i]*=a;
			else
				input[i]/=a;
		__syncthreads();

		// Undo update 2
		a=-0.4435068522;
		if(i%2==0 && i!=0){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==0)
			input[0]+=2*a*input[1];
		__syncthreads();

		// Undo predict 2
		a=-0.8829110762;
		if(i%2==1 && i!=(n-1)){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==(n-1))
			input[n-1]+=2*a*input[n-2];
		__syncthreads();

		// Undo update 1
		a=0.05298011854;
		if(i%2==0 && i!=0){
			input[i]+=a*(input[i-1]+input[i+1]);
		}
		if(i==0)
			input[0]+=2*a*input[1];
		__syncthreads();

		// Undo predict 1
		a=1.586134342;
		if(i%2==1 && i!=(n-1)){
			input[i]+=a*(input[i-1]+input[i+1]);
		} 
		if(i==(n-1))
			input[n-1]+=2*a*input[n-2];
	}
}

#define BLOCK_DIM 16

__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}
