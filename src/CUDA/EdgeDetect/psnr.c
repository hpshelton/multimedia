/*
 *********************************************************************
 *
 *    Computes the PSNR between two images.
 *
 *********************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "array.h"

#define BYTE  unsigned char
#define SQR(x) ((x)*(x))

main(int argc, char *argv[])
{
  int     nrows, ncols, size;
  FILE   *fp1, *fp2;
  BYTE  **img1, **img2;
  int     i, j;
  double  signal=0.0, noise=0.0, peak=0.0, mse;

  if (argc != 5) {
    printf("Usage: %s <nrows> <ncols> <img1> <img2>\n", argv[0]);
    exit(0);
  }
  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  size = nrows * ncols;
  img1  = new_byte_array(nrows, ncols);
  read_byte_array(argv[3], img1, nrows, ncols);
  img2  = new_byte_array(nrows, ncols);
  read_byte_array(argv[4], img2, nrows, ncols);

  for (i=0; i<nrows; i++)
    for (j=0; j<ncols; j++) {
      signal += SQR((double)img1[i][j]);
      noise += SQR((double)img1[i][j] - (double)img2[i][j]);
      if (peak < (double)img1[i][j])
        peak = (double)img1[i][j];
    }

  mse = noise/(double)size;  // Mean square error
  printf("MSE: %lf\n", mse);
  printf("SNR: %lf (dB)\n", 10.0*log10(signal/noise));
  printf("PSNR(max=255): %lf (dB)\n", 10.0*log10(SQR(255.0)/mse));
  printf("PSNR(max=%lf): %lf (dB)\n", peak, 10.0*log10(SQR(peak)/mse));
}




