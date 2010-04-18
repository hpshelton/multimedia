#ifndef ENCODER_H
#define ENCODER_H

#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <QImage>
#include <QString>

#include "utility.h"

class Encoder
{
public:
	//static QImage* test(QImage*);
	static void write_ppc(QImage* image, QString filename, bool huffman, bool arithmetic, bool runlength);
	static unsigned char* huffman_encode(unsigned  char* image, unsigned long* numBytes);
	static unsigned char* runlength_encode(unsigned char* image, unsigned long* numBytes);
	static double* arithmetic_encode(unsigned char* image, unsigned long* numBytes);

private:

	typedef struct Symbol {
		int symbol;
		std::string code;
		float probability;
	};

	class Node
	{
	public:
		std::vector<Symbol*> symbols;
		float probability;

		Node(Symbol* s)
		{
			symbols.push_back(s);
			probability = s->probability;
		}

		Node(Node* first, Node* second)
		{
			probability = 0.0;
			for(unsigned int i = 0; i < first->symbols.size(); i++)
			{
				symbols.push_back(first->symbols[i]);
				probability += first->symbols[i]->probability;
			}
			for(unsigned int i = 0; i < second->symbols.size(); i++)
			{
				symbols.push_back(second->symbols[i]);
				probability += second->symbols[i]->probability;
			}
		}

		void prependToCode(std::string s)
		{
			for(unsigned int i = 0; i < symbols.size(); i++)
				symbols[i]->code = s + symbols[i]->code;
		}
	};

	// Sort by increasing probability
	static struct SymbolsComparator {
		bool operator() (Symbol* i, Symbol* j) { return (i->probability < j->probability); }
	} symbolComparator;

	static struct NodeComparator {
		bool operator() (Node* i, Node* j) { return (i->probability < j->probability); }
	} nodeComparator;
};

#endif // ENCODER_H
