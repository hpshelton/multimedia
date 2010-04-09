#ifndef DEFINES_H
#define DEFINES_H

/** Program information */
#define ORGANIZATION_NAME "Johns Hopkins University"
#define ORGANIZATION_DOMAIN "ece.jhu.edu"
#define PROGRAM_NAME "Multimedia"
#define VERSION "0.1"

#define ZOOM_IN_FACTOR 1.25
#define ZOOM_OUT_FACTOR 0.8

#define CLAMP(a) ( ((a) > 255) ? 255 : (((a) < 0) ? 0 : (int)(a)) )

#endif // DEFINES_H
