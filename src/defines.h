#ifndef DEFINES_H
#define DEFINES_H

/** Program information */
#define ORGANIZATION_NAME "Johns Hopkins University"
#define ORGANIZATION_DOMAIN "ece.jhu.edu"
#define PROGRAM_NAME "Multimedia"
#define VERSION "0.1"

#define QCIF_WIDTH 176
#define QCIF_HEIGHT 144

#define CIF_WIDTH 352
#define CIF_HEIGHT 288

#define ZOOM_IN_FACTOR 1.25
#define ZOOM_OUT_FACTOR 0.8

#define CLAMP(a) ( ((a) > 255) ? 255 : (((a) < 0) ? 0 : (int)(a)) )
#define MIN(a,b) ((a < b) ? a : b)
#define CEIL(a) ( ((a) - (int)(a))==0 ? (int)(a) : (int)((a)+1) )

#define PI 3.14159265358979
#define E 2.71828183

#endif // DEFINES_H
