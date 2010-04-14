/* I had to add 2D free methods to avoid memory leaks */

/* 520.443 Digital Multimedia Coding and Processing */
/* Prof. Trac D. Tran */
/* Department of Electrical and Computer Engineering */
/* The Johns Hopkins University */

#define sat_uchar(t) ((t < 0.0) ? 0.0 : ((t > 255.0) ? 255.0 : t))
#include <stdlib.h>
#include <stdio.h>

FILE *open_get_file (char *message,char *mode,char *dir,char *ext)
{
    char            file[256], temp[256];
    FILE           *fd;

    printf ("%s", message);
    do {
        if(!scanf ("%s", temp))
			return NULL;
        sprintf (file, "%s/%s%s", dir, temp, ext);
        if ((fd = fopen (file, mode)) == NULL) {
	    printf ("\n\t\tError opening file [%s]\n\n", file);
	    printf ("%s", message);
        }
    } while (fd == NULL);

    return (fd);
}

FILE *open_file (const char *file,const char *mode)
{
    FILE           *fd;

    if ((fd = fopen (file, mode)) == NULL)
	printf ("Error opening file [%s]\n\n", file);
    return (fd);
}

/* allocate 2D unsigned char array -- can be used as 1D array (raster scan) */
char **allocate_char (int c,int r)
{
    char          **p;
    int             i;

    if ((p = (char **) malloc (sizeof (char *) * r)) == NULL){
	printf (" Error in space allocation : allocate_char\n");
	exit (0);
    }
    if ((p[0] = (char *)malloc(c*r*sizeof(char))) == NULL){
	printf (" Error in space allocation : allocate_char\n");
	exit (0);
    }
    for (i = 1; i < r; i++) 
	p[i] = p[i-1] + c;
    return p;
}

/* allocate 2D unsigned float array */
float **allocate_float (int c,int r)
{
    float **p;
    int i;

    if ((p = (float **) malloc (sizeof (float *) * r)) == NULL){
		printf (" Error in space allocation : allocate_float\n");
		exit (0);
    }
    if ((p[0] = (float *)malloc(c*r*sizeof(float))) == NULL){
		printf (" Error in space allocation : allocate_float\n");
		exit (0);
    }
    for (i = 1; i < r; i++){ 
		p[i] = p[i-1] + c;
	}
    return p;
}

void free2Dfloat(unsigned char** p)
{
	free(p[0]);
	free(p);
}

/* allocate 2D unsigned unsigned char array */
unsigned char **allocate_uchar (int c,int r)
{
    unsigned char **p;
    int i;

    if ((p = (unsigned char **) malloc (sizeof (unsigned char *) * r)) == NULL){
		printf (" Error in space allocation : allocate_uchar\n");
		exit (0);
    }
    if ((p[0] = (unsigned char *)malloc(c*r*sizeof(unsigned char))) == NULL){
		printf (" Error in space allocation : allocate_uchar\n");
		exit (0);
    }
    for (i = 1; i < r; i++){
		p[i] = p[i-1] + c;
	}
    return p;
}

void free2Duchar(unsigned char** p)
{
	free(p[0]);
	free(p);
}

/* process pgm header */
void get_image_info(FILE *fd,int *row,int *col,int *color)
{
    char P, Five, str[256];

    if(!fread(&P, 1, 1, fd))
		return;
    if(!fread(&Five, 1, 1, fd))
		return;
    rewind(fd);
    if ((P == 'P') && (Five == '5')){
        if(!fgets (str, 256, fd))
			return;
        do {
			if(!fgets (str, 256, fd))
				return;
		} while (str[0] == '#');
        sscanf (str, "%d%d", col, row);
        if(!fgets (str, 256, fd))
			return;
	*color = 1;
    }
    else if ((P == 'P') && (Five == '6')){
        if(!fgets (str, 256, fd))
			return;
        do {
			if(!fgets (str, 256, fd))
				return;
		} while (str[0] == '#');
        sscanf (str, "%d%d", col, row);
        if(!fgets (str, 256, fd))
			return;
	*color = 3;
    }
    else {
	*color = 1;
        if(!fread(col, sizeof(int), 1, fd))
			return;
    	if(!fread(row, sizeof(int), 1, fd))
			return;
    }
}

/* read pgm (raw with simple header) image */
unsigned char **alloc_read_image(char *file,int *row,int *col,int *color)
{
    FILE *fd;
    unsigned char **image;

    if ((fd = fopen(file, "r")) == NULL){
		printf(" Error opening file [%s]\n", file);
		return NULL;
    }
    get_image_info(fd, row, col, color);
    image = allocate_uchar( (*col)*(*color), *row);
    if(!fread(image[0], 1, (*row)*(*col)*(*color), fd))
		return NULL;
    fclose(fd);
    return image;
}

/* write pgm (raw with simple header) image */
/* Typical pgm header for a QCIF frame
   P5
   176 144
   255
*/
void save_image(unsigned char **image,char *file,int row,int col,int color)
{
    FILE *fd;
    
    if ((fd = open_file(file,"w")) == NULL){
	printf("Error opening file [%s]\n", file);
	return ;
    }
    if (color == 3)
        fprintf(fd,"P6\n%d %d\n255\n", col, row);
    else {
        fprintf(fd,"P5\n%d %d\n255\n", col, row);
	color = 1;
    }
    fwrite(image[0], 1, color*row*col, fd);
    fclose(fd);
}
