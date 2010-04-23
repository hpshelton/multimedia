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
	/*// Maps the symbol to its huffman code
	std::map<int, std::string> valueToCode;
	unsigned long num_values = *numBytes;

	// Compute probabilities
	double probabilities[256];
	for(int i = 0; i < 256; i++)
		probabilities[i] = 0.0;

	for(unsigned int i = 0; i < num_values; i++)
		probabilities[image[i]]++;

	for(int i = 0; i < 256; i++)
		probabilities[i] /= num_values;

	// Initialize all the symbols
	std::vector<Symbol*> symbols(256);
	for(int i = 0; i < 256; i++)
	{
		symbols[i] = new Symbol();
		symbols[i]->symbol = i;
		symbols[i]->probability = probabilities[i];
		symbols[i]->code = "";
	}
	sort(symbols.begin(), symbols.end(), symbolComparator);

	// Remove empty probabilities
	unsigned int i = 0;
	while(!symbols[i++]->probability)
		;

	// Create the heap
	std::vector<Node*> nodes;
	for(i = i-1; i < symbols.size(); i++)
		nodes.push_back(new Node(symbols[i]));

	// Combine nodes and prepend the next bit of the code
	while(nodes.size() > 1)
	{
		Node* s1 = nodes[0];
		Node* s2 = nodes[1];
		s1->prependToCode("0");
		s2->prependToCode("1");
		nodes.push_back(new Node(s1, s2));
		nodes.erase(nodes.begin(), nodes.begin() + 2);
		std::sort(nodes.begin(), nodes.end(), nodeComparator);
	}

	// Populate lookup table and write compressed bitstream table
	unsigned char* bitstream = (unsigned char*) malloc(num_values * 8 * sizeof(unsigned char));
	char buffer[8];
	int bufferIndex = 0, bitstreamIndex = 0;

	for(i = 0; i < symbols.size(); i++)
	{
		if(symbols[i]->code.length() > 0)
		{
			std::string no_trailing_zeros(symbols[i]->code);
			while(no_trailing_zeros[no_trailing_zeros.length()-1] == '0' && no_trailing_zeros.length() > 1)
				no_trailing_zeros = no_trailing_zeros.substr(0, no_trailing_zeros.length()-1);
			valueToCode[symbols[i]->symbol] = no_trailing_zeros;

			printf("%d %s %s\n", symbols[i]->symbol, symbols[i]->code.c_str(), no_trailing_zeros.c_str());
			bitstream[bitstreamIndex++] = (unsigned char) symbols[i]->symbol;
			bitstream[bitstreamIndex++] = (unsigned char) ((int) ceil(no_trailing_zeros.length()/8.0));
			for(unsigned int stringIndex = 0; stringIndex < no_trailing_zeros.length(); stringIndex++)
			{
				buffer[bufferIndex++] = no_trailing_zeros[stringIndex];
				if(bufferIndex == 8)
				{
					bufferIndex = 0;
					bitstream[bitstreamIndex++] = (unsigned char)Utility::BintoChar(buffer);
				}
			}
			// Pad with zeroes?
			if(bufferIndex > 0)
			{
				while(bufferIndex < 8)
					buffer[bufferIndex++] = '0';
				bitstream[bitstreamIndex++] = (unsigned char)Utility::BintoChar(buffer);
			}
			bufferIndex = 0;
		}
	}

	for(i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\0';

	// Write encoded symbols
	for(unsigned int i = 0; i < num_values; i++)
	{
		std::string code = valueToCode[image[i]];
		for(unsigned int stringIndex = 0; stringIndex < code.length(); stringIndex++)
		{
			buffer[bufferIndex++] = code[stringIndex];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] = (unsigned char)Utility::BintoChar(buffer);
			}
		}
	}

	// Pad with zeroes?
	if(bufferIndex > 0)
	{
		while(bufferIndex != 8)
			buffer[bufferIndex++] = '0';
		bitstream[bitstreamIndex++] = (unsigned char)Utility::BintoChar(buffer);
	}
	bufferIndex = 0;

	for(i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\0';

	*numBytes = bitstreamIndex;
	return bitstream; */

	int block, i, j;
	int root = 0;
	int index = 0;
	int index2 = 0;
	int sizeOfForest = 256;
	unsigned long hist[sizeOfForest];
	unsigned long totalBlocks = *numBytes;
	float smallest, smallest2;

	tree_node** forest;

	//char** lookup;

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
	/*	for(int i = 0; i < sizeOfForest; i++)
	{
		if(hist[i] > 0.0)
		{
			std::string code(lookup[i]);
			if(code.length() > 0)
			{
				bitstream[bitstreamIndex++] = (unsigned char) i;
//				bitstream[bitstreamIndex++] = (unsigned char) ((int) ceil(code.length()/8.0));
				for(unsigned int b = 0; b < code.length(); b++)
				{
					buffer[bufferIndex++] = code[b];
					if(bufferIndex == 8)
					{
						bufferIndex = 0;
						bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
					}
				}
//				// Pad with zeroes?
//				if(bufferIndex > 0)
//				{
//					while(bufferIndex < 8)
//						buffer[bufferIndex++] = '0';
//					bitstream[bitstreamIndex++] = (unsigned char)Utility::BintoChar(buffer);
//				}
//				bufferIndex = 0;
				buffer[bufferIndex++] = '\n';
				if(bufferIndex == 8)
				{
					bufferIndex = 0;
					bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
				}
			}
		}
	}

	for(i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\0';
	//	fclose(tbl_out);

//	printf("1: %d %d\n", bitstream[0], bitstream[1]);

	//freeTree(forest[root]);
	//free(forest);
*/
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
		printf("%d %d %s ", iterator->first, len, code);
		for(int i = 0; i < len; i++)
		{
			buffer[bufferIndex++] = code[i];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
				printf("%d ", Utility::BintoChar(buffer));
			}
		}
		if(bufferIndex > 0)
		{
			while(bufferIndex < 8)
				buffer[bufferIndex++] = '0';
			bitstream[bitstreamIndex++] = (unsigned char) Utility::BintoChar(buffer);
			bufferIndex = 0;
			printf("%d\n",Utility::BintoChar(buffer));
		}
	}

	for(int i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\n';

	*numBytes = bitstreamIndex;
	/*	// Encode the file using lookup table
	numbits = 0;
	//	rewind(infile);
	//compressed_bitstream = fopen(argv[3],"w");
	Qindex = 0;

	// Write encoded symbols
	for(unsigned int i = 0; i < *numBytes; i++)
	{
		block = image[i];
		//printf("%d ", block);
		j = 0;
		while(lookup[block][j] != '\0')
		{
			Iqueue[Qindex++] = lookup[block][j];
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
		numbits += Utility::numBits(lookup[block]);
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

	// Free all memory
	//for(i=0; i < 256; i++)
		//free(lookup[i]);
//	free(lookup);

	*numBytes = bitstreamIndex;
	return bitstream;*/
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

	printf("%d\n", codeToValue[std::string("10")]);
/*	// Read in the symbols
	unsigned char* image = (unsigned char*) malloc(*numBytes * 8 * sizeof(unsigned char));
	int index = 0;
	code = new std::string();
	while(bitstream[bitstreamIndex] != '\0' || bitstream[bitstreamIndex+1] != '\0' || bitstream[bitstreamIndex+2] != '\0')
	{
		bits = Utility::getBinVal((int) bitstream[bitstreamIndex++], 8);
		for(int b = 0; b < 8; b++)
		{
			code->append(1, bits[b]);
			std::string padded(*code);
			int value;
			int count = 0;
			while(padded.length() % 8 != 0)
				padded.append(1, '0');
			std::map<std::string, int>::iterator v = codeToValue.find(padded);
			if(v != codeToValue.end())
			{
				value = v->second;
				printf("%d\n", (int) value);
				image[index++] = (unsigned char) value;
				code = new std::string();
			}
		}
	}

	*numBytes = index;
	return image;*/

/*	tree_node* root;
	int sym;
	int newVal = 1;
	char c = 1;
	char* BPP;
	int i = 0;

	int byte;
	char* bits;
	char bit;

	int bitstreamIndex = 0;
	tree_node* current;

	// Parse in the huffman_table, build the lookup tree
	root = (tree_node*) malloc(sizeof(tree_node));

	// Read in the Huffman table
	while(bitstream[bitstreamIndex] != '\0' || bitstream[bitstreamIndex+1] != '\0' || bitstream[bitstreamIndex+2] != '\0')
	{
		sym = (int) bitstream[bitstreamIndex++];
		//		int	bytes = (int) bitstream[bitstreamIndex++];
		char* byte2 = Utility::charToBin(bitstream[bitstreamIndex++]);
		printf("%s\n", byte2);
		int byte_index = 0;
		BPP = (char*) malloc(sizeof(char) * 32);
		while(byte2[byte_index] != '\n')
		{
			BPP[i++] = (char)byte2[byte_index++];
			if(byte_index == 8)
			{
				byte_index = 0;
				byte2 = Utility::charToBin(bitstream[bitstreamIndex++]);
			}
		}

		//		BPP = (char*) malloc(sizeof(char) * 32);
		//	if(fscanf(huff_tbl, "%d\t", &sym)!=EOF){
		//	}
		//	else{
		//		newVal=0;
		//	}

		//while( (c=fgetc(huff_tbl)) != '\n' && c!=EOF){
	//		BPP[i++]=c;
		//}
		BPP[i] = '\0';
		printf("%d %s: ", sym, BPP);
		i = 0;
		addToTree(root, BPP, sym);
		free(BPP);
	}*/
	/*
	// Read in the compressed file, convert it into binary, decode the binary, and write it out
	current = root;
	while(bitstream[bitstreamIndex] != '\0' || bitstream[bitstreamIndex+1] != '\0' || bitstream[bitstreamIndex+2] != '\0')
	{
		bits = Utility::charToBin(bitstream[bitstreamIndex++]);
		i=0;
		while((bit = bits[i]) != '\0' && bit != 0)
		{
			if(bit == '0')
			{
				if(current->left)
					current = current->left;
				if(!current->left && !current->right)
				{
					printf("%d", current->sym);
					current = root;
				}
			}
			else if(bit == '1')
			{
				if(current->right)
					current=current->right;
				if(!current->left && !current->right)
				{
					printf("%d", current->sym);
					current = root;
				}
			}
			i++;
		}
		free(bits);
	}

	// Free all memory

	freeTree(root);
	//fclose(huff_tbl);
	//fclose(cbs);
//	fclose(out);
	return 0; */
}
