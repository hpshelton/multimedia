#include "encoder.h"
#include "decoder.h"

typedef struct tree_node{
	float prob;
	int sym;
	char* code;
	struct tree_node *right, *left;
};

tree_node* initTree(tree_node* a, tree_node* b)
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
	newNode->prob = prob;
	newNode->sym = sym;
	newNode->left = NULL;
	newNode->right = NULL;
	return newNode;
}

void assign_codes(tree_node* root, int val, long int digits, std::map<int, char*> &lookup)
{
	char* binvalstr;
	if(root->sym >= 0 && root->prob > 0)
	{
		val /= 2;
		binvalstr = Utility::getBinVal(val, digits);
		root->code = binvalstr;
		lookup[root->sym] = binvalstr;
	}
	if(root->left)
		assign_codes(root->left, 2*val, digits+1, lookup);
	if(root->right)
		assign_codes(root->right, 2*(val+1), digits+1, lookup);
}

void freeTree(tree_node* root)
{
	if(root->left)
		freeTree(root->left);
	if(root->right)
		freeTree(root->right);
	free(root);
}

void addToTree(tree_node* root, char* path, int sym)
{
	char c;
	tree_node* current = root;
	int i = 0;
	while((c = path[i++]) != '\0')
	{
		if(c == '0')
		{
			if(current->left)
				current = current->left;
			else
			{
				current->left = initTree(1,-1);
				current=current->left;
			}
		}
		else if(c == '1')
		{
			if(current->right)
				current = current->right;
			else
			{
				current->right = initTree(1,-1);
				current = current->right;
			}
		}
	}
	current->sym = sym;
}

unsigned char* Encoder::huffman_encode(unsigned char* image, unsigned long* numBytes)
{
	int num_symbols = 256;
	unsigned long hist[num_symbols];
	tree_node** forest = (tree_node**) malloc(num_symbols * sizeof(tree_node*));

	// Compute probabilities
	for(int i = 0; i < num_symbols; i++)
		hist[i] = 0.0;

	for(unsigned int i = 0; i < *numBytes; i++)
		hist[image[i]]++;

	for(int i = 0; i < num_symbols; i++)
	{
		if(hist[i])
			forest[i] = initTree(((float)hist[i])/(*numBytes), i);
		else
			forest[i] = 0;
	}

	// Compute Huffman codes
	int index = 0, index2 = 0, root = 0;
	for(int i = 0; i < num_symbols-1; i++)
	{
		float smallest = 1, smallest2 = 1;
		for(int j = 0; j < num_symbols; j++)	// Find the two smallest probabilities
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

		// Combine the two smallest probabilities
		forest[index] = initTree(forest[index], forest[index2]);
		forest[index2] = 0;
		root = index;
	}

	// Save the Huffman codes in a lookup table
	std::map<int, char*> lookup;
	assign_codes(forest[root], 0, 0, lookup);

	// Write out the table
	unsigned char* bitstream = (unsigned char*) malloc(*numBytes * sizeof(unsigned char));
	char buffer[8];
	int bufferIndex = 0, bitstreamIndex = 0;

	for(std::map<int,char*>::iterator iterator = lookup.begin(); iterator != lookup.end(); iterator++)
	{
		bitstream[bitstreamIndex++] = iterator->first;
		char* code = iterator->second;
		int len = std::string(code).length();
		if(len > 255)
		{
			std::cerr << "Huffman code symbol length exceeds 255 bits" << std::endl;
			exit(1);
		}
		bitstream[bitstreamIndex++] = (unsigned char) len;
		for(int i = 0; i < len; i++)
		{
			buffer[bufferIndex++] = code[i];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
			}
		}
		if(bufferIndex > 0)
		{
			while(bufferIndex < 8)
				buffer[bufferIndex++] = '0';
			bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
			bufferIndex = 0;
		}
	}

	// Terminate the table
	for(int i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\n';

	// Write out the file using the lookup table
	for(unsigned int i = 0; i < *numBytes; i++)
	{
		int j = 0;
		unsigned char byte = image[i];
		char* code = lookup[(int)byte];
		while(code[j] != '\0')
		{
			buffer[bufferIndex++] = code[j++];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] = Utility::BintoChar(buffer);
			}
		}
	}

	// Pad the output with zeros
	if(bufferIndex > 0)
	{
		while(bufferIndex < 8)
			buffer[bufferIndex++] = '0';
		bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
		bufferIndex = 0;
	}

	freeTree(forest[root]);
	*numBytes = bitstreamIndex;
	return bitstream;
}

unsigned char* Decoder::huffman_decode(unsigned char* bitstream, unsigned long* numBytes)
{
	std::map<std::string, int> codeToValue;
	unsigned int bitstreamIndex = 0;
	char* bits;
	char* symbol_code;

	// Read in the Huffman table
	while(bitstream[bitstreamIndex] != '\n' || bitstream[bitstreamIndex+1] != '\n' || bitstream[bitstreamIndex+2] != '\n')
	{
		int symbol = (int) bitstream[bitstreamIndex++];
		int num_bits = (int) bitstream[bitstreamIndex++];
		bits = Utility::charToBin(bitstream[bitstreamIndex++]);
		symbol_code = (char*) malloc((num_bits+1)*sizeof(char));
		int bit_index = 0;
		int code_index = 0;
		while(code_index < num_bits)
		{
			symbol_code[code_index++] = bits[bit_index++];
			if(bit_index == 8 && num_bits > 8)
			{
				bits = Utility::charToBin(bitstream[bitstreamIndex++]);
				bit_index = 0;
			}
		}
		symbol_code[code_index] = '\0';
		codeToValue[std::string(symbol_code)] = symbol;
	}
	bitstreamIndex += 3;

	// Read in the symbols
	unsigned char* image = (unsigned char*) malloc(ceil(*numBytes * 1.5) * sizeof(unsigned char));
	int image_index = 0;
	std::string* code = new std::string();
	while(bitstreamIndex < *numBytes)
	{
		char* byte = Utility::getBinVal((int) bitstream[bitstreamIndex++], 8);
		for(int b = 0; b < 8; b++)
		{
			// Append one bit and check for symbol in the table
			code->append(1, byte[b]);
			std::map<std::string, int>::iterator v = codeToValue.find(*code);
			if(v != codeToValue.end())
			{
				image[image_index++] = (unsigned char) v->second;
				code = new std::string();
			}
		}
	}

	delete bits;
	delete code;
	delete symbol_code;

	*numBytes = image_index;
	return image;
}
