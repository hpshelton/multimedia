#define CLAMP(a) ((a>255) ? 255 : ((a<0) ? 0 : a))

__global__ void edge_detect(unsigned char* input, unsigned char* output, int row, int col)
{
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	int index = xIndex + yIndex * row;

	if(index < row*col){
		float coeff[3][3]= {{-1, -1, -1},
							{-1,  8, -1},
							{-1, -1, -1}};

		int i, j;
		float convSum=0;
		for(i=-1; i < 2; i++){
			for(j=-1; j < 2; j++){
				if(-1 < (index+j)+(row*i) && (index+j)+(row*i) < row*col){
					convSum += coeff[i+1][j+1]*input[(index+j)+(row*i)];
				}
			}
		}
		output[index] = CLAMP(convSum);
	}
}
