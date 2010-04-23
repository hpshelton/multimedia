#include "encoder.h"
#include "decoder.h"

typedef struct tree_node{
	float prob;
	int sym;
	char* code;
	struct tree_node *right, *left;
};

tree_node* makeNewTree(tree_node* a, tree_node* b)
{
	tree_node* newNode = (tree_node*) malloc(sizeof(tree_node));
	newNode->prob = a->prob + b->prob;
	newNode->sym = -1;
	newNode->left = a;
	newNode->right= b;
	return newNode;
}

tree_node* initTree(float prob, int sym)
{
	tree_node* newNode = (tree_node*) malloc(sizeof(tree_node));
	newNode->prob= prob;
	newNode->sym = sym;
	newNode->left =0;
	newNode->right=0;
	return newNode;
}

void fprint_ascii_codes(FILE* output, tree_node* root, int val, long int digits, std::map<int, char*> &lookup)
{
	char* binvalstr;
	if(root->sym >= 0 && root->prob > 0)
	{
		val = val/2;
		binvalstr = Utility::getBinVal(val, digits);
		root->code = binvalstr;
		lookup[root->sym] = binvalstr;
		//printf("ASCII: %d %s\n", root->sym, lookup[root->sym]);
	}
	if(root->left){
		fprint_ascii_codes(output, root->left, val*2, 1+digits, lookup);
	}
	if(root->right){
		fprint_ascii_codes(output, root->right, (1+val)*2, 1+digits, lookup);
	}
}

void freeTree(tree_node* root)
{
	if( root->left){
		freeTree(root->left);
	}
	if( root->right){
		freeTree(root->right);
	}
	free(root);
}

void addToTree(tree_node* root, char* path, int sym)
{
	char c;
	tree_node* current = root;
	int i=0;
	while((c = path[i++]) != '\0')
	{
		if(c == '0')
		{
			if(current->left)
				current=current->left;
			else
			{
				current->left = initTree(1,-1);
				current=current->left;
			}
		}
		else if(c == '1')
		{
			if(current->right){
				current=current->right;
			}
			else{
				current->right = initTree(1,-1);
				current=current->right;
			}
		}
	}
	current->sym = sym;
}

unsigned char* Encoder::huffman_encode(unsigned char* image, unsigned long* numBytes)
{
	int block, i, j;
	int root = 0;
	int index = 0;
	int index2 = 0;
	int sizeOfForest = 256;
	unsigned long hist[sizeOfForest];
	unsigned long totalBlocks = *numBytes;
	float smallest, smallest2;

	tree_node** forest;


	int numbits;
	char byte[8];
	int k;
	int Qindex;
	int Iqueue[8];

	// Compute probabilities
	for(int i = 0; i < 256; i++)
		hist[i] = 0.0;

	for(unsigned int i = 0; i < *numBytes; i++)
		hist[image[i]]++;

	// Compute the Huffman Codes
	forest = (tree_node**) malloc(sizeOfForest * sizeof(tree_node*));
	for(i = 0; i < sizeOfForest; i++)
	{
		if(hist[i])
			forest[i] = initTree(((float)hist[i])/totalBlocks, i);
		else
			forest[i] = 0;
	}

	for(i = 0; i < sizeOfForest-1; i++)	// for the whole build
	{
		smallest = 1;
		smallest2 = 1;
		for(j = 0; j < sizeOfForest; j++)	// find the 2 smallest probs
		{
			if(forest[j])
			{
				if(forest[j]->prob <= smallest)
				{
					smallest2 = smallest;
					index2 = index;
					smallest = forest[j]->prob;
					index = j;
				}
				else if(forest[j]->prob <= smallest2)
				{
					smallest2 = forest[j]->prob;
					index2 = j;
				}
			}
		}
		if(smallest == 1)
			break;

		//combine the two smallest probs
		forest[index] = makeNewTree(forest[index], forest[index2]);
		forest[index2] = 0;
		root = index;
	}

	// Print the Huffman codes, save them in a lookup table
	std::map<int, char*> lookup;
	unsigned char* bitstream = (unsigned char*) malloc(*numBytes * 8 * sizeof(unsigned char));
	char buffer[8];
	int bufferIndex = 0, bitstreamIndex = 0;

	//	tbl_out = fopen(argv[2], "w");
	fprint_ascii_codes(NULL, forest[root], 0, 0, lookup);

	for(std::map<int,char*>::iterator iterator = lookup.begin(); iterator != lookup.end(); iterator++)
	{
		bitstream[bitstreamIndex++] = iterator->first;
		char* code = iterator->second;
		int len = std::string(code).length();
		if(len > 255)
		{
			printf("Huffman code symbol length exceeds 255 bits\n");
			exit(1);
		}
		bitstream[bitstreamIndex++] = (unsigned char) len;
	//	printf("%d %d %s ", iterator->first, len, code);
		for(int i = 0; i < len; i++)
		{
			buffer[bufferIndex++] = code[i];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
		//		printf("%d ", Utility::BintoChar(buffer));
			}
		}
		if(bufferIndex > 0)
		{
			while(bufferIndex < 8)
				buffer[bufferIndex++] = '0';
			bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
			bufferIndex = 0;
	//		printf("%d\n",Utility::BintoChar(buffer));
		}
	}

	for(int i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\n';

	//*numBytes = bitstreamIndex;
	// Encode the file using lookup table
	numbits = 0;
	//	rewind(infile);
	//compressed_bitstream = fopen(argv[3],"w");
	Qindex = 0;

	// Write encoded symbols
	for(unsigned int i = 0; i < *numBytes; i++)
	{
		unsigned char block2 = image[i];
		//printf("%d ", block);
		j = 0;
		char* code = lookup[(int)block2];
		while(code[j] != '\0')
		{
			Iqueue[Qindex++] = code[j];
			if(Qindex == 8)
			{
				Qindex = 0;
				for(k = 0; k < 8; k++)
				{
					byte[k] = Iqueue[k];
				}
			//	printf("%d ", Utility::BintoChar(byte));
				bitstream[bitstreamIndex++] = Utility::BintoChar(byte);
			}
			j++;
		}
		numbits += Utility::numBits(code);
//		printf("\n");
	}

	while(numbits % 8)
	{
		Iqueue[Qindex++] = '0';
		if(Qindex==8)
		{
			Qindex=0;
			for(k=0; k < 8; k++)
			{
				byte[k] = Iqueue[k];
			}
		//	printf("%d\n", Utility::BintoChar(byte));
			bitstream[bitstreamIndex++] = Utility::BintoChar(byte);
		}
		numbits++;
	}

	// TODO - Free all memory
	*numBytes = bitstreamIndex;
	return bitstream;
}

unsigned char* Decoder::huffman_decode(unsigned char* bitstream, unsigned long* numBytes)
{
	std::map<std::string, int> codeToValue;
	int symbol;
	char* bits;
	int bytes;
	std::string* code;
	int bitstreamIndex = 0;

	// Read in the Huffman table
	while(bitstream[bitstreamIndex] != '\n' || bitstream[bitstreamIndex+1] != '\n' || bitstream[bitstreamIndex+2] != '\n')
	{
		symbol = (int) bitstream[bitstreamIndex++];
		int num_bits = (int) bitstream[bitstreamIndex++];
		char* bits = Utility::charToBin(bitstream[bitstreamIndex++]);
		int bit_index = 0;
		char* code = (char*) malloc((num_bits+1)*sizeof(char));
		int len = 0;
		while(len < num_bits)
		{
			code[len++] = bits[bit_index++];
			if(bit_index == 8 && num_bits > 8)
			{
				bits = Utility::charToBin(bitstream[bitstreamIndex++]);
				bit_index = 0;
			}
		}
		code[len] = '\0';
		codeToValue[std::string(code)] = symbol;
	}
	bitstreamIndex += 3;

	// Read in the symbols
	unsigned char* image = (unsigned char*) malloc(*numBytes * 8 * sizeof(unsigned char));
	int index = 0;
	code = new std::string();
	while(bitstreamIndex < *numBytes)
	{
		bits = Utility::getBinVal((int) bitstream[bitstreamIndex++], 8);
		for(int b = 0; b < 8; b++)
		{
			code->append(1, bits[b]);
			int value;
			std::map<std::string, int>::iterator v = codeToValue.find(*code);
			if(v != codeToValue.end())
			{
				value = v->second;
	//			printf("%d\n", (int) value);
				image[index++] = (unsigned char) value;
				code = new std::string();
			}
		}
	}

	*numBytes = index;
	return image;

	// TODO - Free all memory
}
