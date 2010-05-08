#define CLAMP(a) ((a>255) ? 255 : ((a<0) ? 0 : (unsigned char)(a)))

__global__ void setToVal(unsigned char* x, int len, int val)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if(index < len)
		x[index] = val;
}

__global__ void setToVal(float* x, int len, int val)
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

__global__ void shuffle(float* output, unsigned char* input, int width, int height)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<width*height*4)
		output[i/4 + (i%4)*width*height] = input[i];
}
__global__ void UNshuffle(unsigned char* output, float* input, int width, int height)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<width*height*4)
		output[i] = CLAMP(input[i/4 + (i%4)*width*height]);
}

#define ABS(a) (a<0?-a:a)


__global__ void zeroOut(int* x, float threshold, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<n){
		if(ABS(x[i]) < threshold)
			x[i]=0;
	}
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

__global__ void contrast(unsigned char* input, unsigned char* output, int row, int col, float factor, float lum)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = (4*xIndex + yIndex * row*4);

	if(index < row*col*4){
		output[index]=CLAMP( (1-factor)*lum + factor*input[index] );
		output[index+1]=CLAMP( (1-factor)*lum + factor*input[index+1] );
		output[index+2]=CLAMP( (1-factor)*lum + factor*input[index+2] );
		output[index+3]=0;
	}
}

__global__ void predict(float* input, int n, int len, int dim, float a, int width, int height)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int color = threadIdx.z*(n/4);
	int i = (xIndex + yIndex * dim);

	if (xIndex < width && yIndex < height && i+color < n)
	{
		if(i%2==1){
			if((i%len)!=(len-1)){
				input[color+i]+=a*(input[color+i-1]+input[color+i+1]);
			}
			else{
				input[color+i]+=2*a*input[color+i-1];
			}
		}
	}
}

__global__ void update(float* input, int n, int len, int dim, float a, int width, int height)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int i = (xIndex + yIndex * dim);
	int color = threadIdx.z*(n/4);

	if (xIndex < width && yIndex < height && i+color < n)
	{
		if(i%2==0){
			if((i%len)!=0 && (i%len)!=len-1){
				input[color+i]+=a*(input[color+i-1]+input[color+i+1]);
			}
			else if(i%len==0){
				input[color+i]+=2*a*input[color+i+1];
			}
			else{
				input[color+i]+=2*a*input[color+i-1];
			}
		}
	}
}

__global__ void scale(float* input, int n, int len, int dim, float a, int width, int height)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex < width && yIndex < height && (xIndex + yIndex * dim) + threadIdx.z*(n/4) < n)
	{
	int i = (xIndex + yIndex * dim) + threadIdx.z*(n/4);
		if (i%2)
			input[i] = input[i]*a;
		else
			input[i] = input[i]/a;
	}
}

__global__ void pack(float* input, float* tempbank, int n, int len, int dim)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int i = (xIndex + yIndex * dim);
	int color = threadIdx.z*(n/4);

	if ((i+color)<n)
	{
		int rowNum = i / len;
		int rowIndex = i%len;
		if (i%2==0)
			tempbank[color + len*rowNum + rowIndex/2] = input[color+i];
		else
			tempbank[color + len*rowNum + rowIndex/2 + len/2] = input[color+i];
	}
}

__global__ void unpack(float* input, float* tempbank, int n, int len, int dim)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int i = (xIndex + yIndex * dim);
	int color = threadIdx.z*(n/4);

	if ((i+color)<n)
	{
		int rowNum = i / len;
		int rowIndex = i%len;
		if(i%2==0)
			tempbank[color+i] = input[color + len*rowNum + rowIndex/2];
		else
			tempbank[color+i] = input[color + len*rowNum + rowIndex/2 + len/2];
	}
}

__global__ void readOut(float* input, float* tempbank, int n, int len, int dim)
{
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int i = (xIndex + yIndex * dim)+threadIdx.z*(n/4);;

	if ((i)<n)
		input[i]=tempbank[i];
}

__global__ void roundArray(int* output, float* input, int width, int height)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<width*height*4)
		output[i] = (int)(input[i]+0.5);
}

__global__ void intToFloat(float* output, int* input, int width, int height)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<width*height*4)
		output[i] = input[i];
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
