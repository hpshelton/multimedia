#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

using namespace std;

int main(int argc, char* argv[])
{
	if(argc != 3){
		printf("usage: %s infile outfile\n",argv[0]);
		return -1;
	}

	FILE* infile = fopen(argv[1],"r");
	FILE* outfile = fopen(argv[2],"w");

	vector<int> xVecs;
	vector<int> yVecs;
	vector<int> diffs;

	int xVec, yVec, diff;
	while(fscanf(infile, "%d\t%d\t%d\n",&xVec, &yVec, &diff)!=EOF)
	{
		xVecs.push_back(xVec);
		yVecs.push_back(yVec);
		diffs.push_back(diff);
	}

	int xstart = 4;
	fprintf(outfile,"X = [");
	for(int i=0; i < xVecs.size(); i++){
		if(xstart > 176)
			xstart=4;
		fprintf(outfile, " %d ",xstart);
		xstart+=8;		
	}
	fprintf(outfile,"];\n");

	int ystart = 4;
	fprintf(outfile,"Y = [");
	for(int i=0; i < xVecs.size(); i++){
		fprintf(outfile, " %d ",ystart);
		if(!(i%(int)(ceil(176/8))) && i!=0)
			ystart+=8;
	}
	fprintf(outfile,"];\n");

	fprintf(outfile,"U = [");
	for(int i=0; i < xVecs.size(); i++){
		fprintf(outfile, " %d ",xVecs[i]);
	}
	fprintf(outfile,"];\n");

	fprintf(outfile,"V = [");
	for(int i=0; i < yVecs.size(); i++){
		fprintf(outfile, " %d ",yVecs[i]);
	}
	fprintf(outfile,"];\n");

	fprintf(outfile,"figure;\nimshow('frame2.png');\nhold on;\nquiver(X,Y,U,V,0);");

	fclose(infile);
	fclose(outfile);
	return 0;
}
