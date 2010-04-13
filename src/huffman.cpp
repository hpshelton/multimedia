#include "encoder.h"
#include "decoder.h"

#include <iostream>

unsigned char* Encoder::huffman_encode(unsigned char* image, unsigned int num_values, unsigned int* numBytes)
{
	// Maps the symbol to its huffman code
	std::map<int, std::string> valueToCode;

	// Compute probabilities
	float probabilities[256];
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
	while(nodes.size() >= 2)
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
	unsigned char* bitstream = (unsigned char*) malloc(num_values*8*sizeof(unsigned char));
	char buffer[8];
	int bufferIndex = 0, bitstreamIndex = 0;

	for(i = 0; i < symbols.size(); i++)
	{
		if(symbols[i]->code.length() > 0)
		{
			valueToCode[symbols[i]->symbol] = symbols[i]->code;
			bitstream[bitstreamIndex++] = (unsigned char) symbols[i]->symbol;
			bitstream[bitstreamIndex++] = (unsigned char) ((int)ceil(symbols[i]->code.length()/8.0));
			for(unsigned int stringIndex = 0; stringIndex < symbols[i]->code.length(); stringIndex++)
			{
				buffer[bufferIndex++] = symbols[i]->code[stringIndex];
				if(bufferIndex == 8)
				{
					bufferIndex = 0;
					bitstream[bitstreamIndex++] += (unsigned char)Utility::BintoChar(buffer);
				}
			}
			// Pad with zeroes?
			if(bufferIndex > 0)
			{
				while(bufferIndex != 8)
					buffer[bufferIndex++] = '0';
				bitstream[bitstreamIndex++] += (unsigned char)Utility::BintoChar(buffer);
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
		bitstream[bitstreamIndex++] = (unsigned char) ((int)ceil(code.length()/8.0));
		for(unsigned int stringIndex = 0; stringIndex < code.length(); stringIndex++)
		{
			buffer[bufferIndex++] = code[stringIndex];
			if(bufferIndex == 8)
			{
				bufferIndex = 0;
				bitstream[bitstreamIndex++] += (unsigned char)Utility::BintoChar(buffer);
			}
		}
		// Pad with zeroes?
		if(bufferIndex > 0)
		{
			while(bufferIndex != 8)
				buffer[bufferIndex++] = '0';
			bitstream[bitstreamIndex++] += (unsigned char)Utility::BintoChar(buffer);
		}
		bufferIndex = 0;
	}

	for(i = 0; i < 3; i++)
		bitstream[bitstreamIndex++] = '\0';

	*numBytes = bitstreamIndex;
	return bitstream;
}

unsigned char** Decoder::huffman_decode(unsigned char* bitstream, int width, int height)
{
	std::map<std::string, int> codeToValue;
	int symbol;
	int bytes;
	std::string* code;
	int bitstreamIndex = 0;

	// Read in the Huffman table
	while(bitstream[bitstreamIndex] != '\0' || bitstream[bitstreamIndex+1] != '\0' || bitstream[bitstreamIndex+2] != '\0')
	{
		symbol = (int)bitstream[bitstreamIndex++];
		bytes = (int)bitstream[bitstreamIndex++];
		code = new std::string();
		for(int b = 0; b < bytes; b++)
			code->append(Utility::getBinVal((int)bitstream[bitstreamIndex++], 8));
		codeToValue[*code] = symbol;
	}
	bitstreamIndex += 3;

	// Read in the symbols
	unsigned char* image = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
	int index = 0;
	while(bitstream[bitstreamIndex] != '\0' || bitstream[bitstreamIndex+1] != '\0' || bitstream[bitstreamIndex+2] != '\0')
	{
		bytes = (int)bitstream[bitstreamIndex++];
		code = new std::string();
		for(int b = 0; b < bytes; b++)
			code->append(Utility::getBinVal((int)bitstream[bitstreamIndex++], 8));
		image[index++] = codeToValue[*code];
	}

	return Utility::blockArray(image, height*3, width);
}
