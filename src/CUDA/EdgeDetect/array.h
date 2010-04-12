/*
 ***************************************************************************
 *
 *   Various functions for manipulating dynamically allocated 2D arrays.
 *
 *
 *   - By Yuan-Liang Tang, Last modified 2/15/2005
 *
 ***************************************************************************
 */
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#define BYTE unsigned char

// Allocate arrays
#define  new_byte_array(nr, nc) (BYTE **)new_array(sizeof(BYTE), (nr), (nc))
#define  new_int_array(nr, nc) (int **)new_array(sizeof(int), (nr), (nc))
#define  new_flt_array(nr, nc) (float **)new_array(sizeof(float), (nr), (nc))
#define  new_double_array(nr, nc) (double **)new_array(sizeof(double), (nr), (nc))

// Read array files
#define  read_byte_array(fn, ptr, nr, nc)  read_array((fn), (char **)(ptr), sizeof(BYTE), (nr), (nc))
#define  read_int_array(fn, ptr, nr, nc)  read_array((fn), (char **)(ptr), sizeof(int), (nr), (nc))
#define  read_flt_array(fn, ptr, nr, nc)  read_array((fn), (char **)(ptr), sizeof(float), (nr), (nc))
#define  read_double_array(fn, ptr, nr, nc)  read_array((fn), (char **)(ptr), sizeof(double), (nr), (nc))

// Write array files
#define  write_byte_array(fn, ptr, nr, nc)  write_array((fn), (char **)(ptr), sizeof(BYTE), (nr), (nc))
#define  write_int_array(fn, ptr, nr, nc)  write_array((fn), (char **)(ptr), sizeof(int), (nr), (nc))
#define  write_flt_array(fn, ptr, nr, nc)  write_array((fn), (char **)(ptr), sizeof(float), (nr), (nc))
#define  write_double_array(fn, ptr, nr, nc)  write_array((fn), (char **)(ptr), sizeof(double), (nr), (nc))

/*
 ***************************************************************************
 *
 *    Create a new array.
 *
 *    Examples of invocation:
 *       int **arr1;
 *       double **arr2;
 *       unsigned char **arr3;
 *       arr1 = (int **)new_array(sizeof(int), 256, 256);
 *       arr2 = (double **)new_array(sizeof(double), 256, 256);
 *       arr3 = (unsigned char **)new_array(sizeof(unsigned char), 256, 256);
 *
 ***************************************************************************
 */
char  **new_array(int unit_size, int nrows, int ncols)
{
  char  **arr;
  int     i;

  if (!(arr=(char **)malloc(nrows*sizeof(char *)))) {
    printf("In 'new_array()': memory allocation error.\n");
    exit(0);
  }
  if (!(arr[0]=(char *)malloc(nrows*ncols*unit_size))) {
    printf("In 'new_array()': memory allocation error.\n");
    exit(0);
  }
  for (i=1; i<nrows; i++)
    arr[i] = arr[i-1] + ncols*unit_size;
  return arr;
}


/*
 ***************************************************************************
 *
 *    Read an array from a file.
 *
 *    Examples of invocation:
 *       char    filename[100];
 *       unsigned char   **arr1;
 *       int   **arr2;
 *       double  **arr3;
 *       read_array(filename, (char **)arr1, sizeof(unsigned char), 256, 256);
 *       read_array(filename, (char **)arr2, sizeof(int), 256, 256);
 *       read_array(filename, (char **)arr3, sizeof(double), 256, 256);
 *
 ***************************************************************************
 */
void read_array(char *filename, char **array, int unit_size,
		int nrows, int ncols)
{
  FILE  *fp;

  if (!(fp=fopen(filename, "rb"))) {
    printf("In 'read_array()': cannot read from '%s'\n", filename);
    exit(0);
  }
  fread(array[0], unit_size, nrows*ncols, fp);
  fclose(fp);
}


/*
 ***************************************************************************
 *
 *    Write an array to a file.
 *
 *    Examples of invocation:
 *       char    filename[100];
 *       unsigned char **arr1;
 *       int   **arr2;
 *       double   **arr3;
 *       write_array(filename, (char **)arr1, sizeof(unsigned char), 256, 256);
 *       write_array(filename, (char **)arr2, sizeof(int), 256, 256);
 *       write_array(filename, (char **)arr3, sizeof(double), 256, 256);
 *
 ***************************************************************************
 */
void write_array(char *filename, char **array, int unit_size,
		 int nrows, int ncols)
{
  FILE  *fp;

  if (!(fp=fopen(filename, "wb"))) {
    printf("In 'write_array()': cannot write to '%s'\n", filename);
    exit(0);
  }
  fwrite(array[0], unit_size, nrows*ncols, fp);
  fclose(fp);
}

