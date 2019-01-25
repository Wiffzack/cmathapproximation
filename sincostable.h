#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
//#include "fastlog.h"
#include <stdint.h>

#include <emmintrin.h>
#include <xmmintrin.h>
#include <x86intrin.h>


#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

//float to int
#define FLOAT_FTOI_MAGIC_NUM (float)(3<<21)
#define IT_FTOI_MAGIC_NUM (0x4ac00000)

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif



#define M_LOG2E 1.4426950408889634074
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144  
#define I_PI	    0.31830988618379067153776752674502872
#define Eul	    2.7182818284590452353602874

#define MAX_CIRCLE_ANGLE      512
#define HALF_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/2)
#define QUARTER_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/4)
#define MASK_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE - 1)


typedef union
{
    double   f;
    uint64_t ui;
    int64_t  si;
} fi_t;

static const uint64_t fastlog_exp_mask = 0x7ff0000000000000;
static const uint64_t fastlog_man_mask = 0x000fffffffffffff;


double* fastlog_lookup  = NULL;
uint64_t fastlog_man_offset = 0;





//extern double sin ( double );

#pragma pack(push,1)
typedef struct triglist2{
        float fast_cossin_table[MAX_CIRCLE_ANGLE] __attribute__ ((aligned (16)));
};
#pragma pack(pop)
struct triglist2 lib1;

// convert float to int (int)
int convert5(float x)
{
    return _mm_cvtt_ss2si(_mm_load_ss(&x)); // extra 't' means truncate
}

// int to float (float)
int convert6(float f)
{
	f += FLOAT_FTOI_MAGIC_NUM;
	return (*((int*)&f) - IT_FTOI_MAGIC_NUM)>>1;
}




void InitSinTable1()
{
        unsigned int i;
        #pragma omp parallel
        #pragma omp for
        for (i = 0 ; i < MAX_CIRCLE_ANGLE ; i++)
        {
                lib1.fast_cossin_table[i] = (float)sinf((double)i * M_PI / HALF_MAX_CIRCLE_ANGLE);
        }
}

double cos(double n)
{
   if(unlikely(lib1.fast_cossin_table[45]==0))
   {
   InitSinTable1();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&lib1.fast_cossin_table[0],1,1);
   d = lib1.fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
        return lib1.fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   }
   return d;
}

double sin(double n)
{
   if(unlikely(lib1.fast_cossin_table[45]==0))
   {
   InitSinTable1();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&lib1.fast_cossin_table[0],1,1);
   d = lib1.fast_cossin_table[i&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
      return lib1.fast_cossin_table[(-((-i)&MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE];
   }
   return d;
}

double tan(double x)
{
	return sin(x)/cos(x);
}

double cot(double x)
{
	return (cos(x)/sin(x));
}

void fastlog_init(int prec)
{
    if (prec < 1 || prec > 52) {
        abort();
    }

    free(fastlog_lookup);

    uint64_t n = 1 << prec; // 2^prec
    fastlog_lookup = malloc(n * sizeof(double));

    if (fastlog_lookup == NULL) {
        abort();
    }

    fastlog_man_offset = 52 - prec;
    uint64_t x;
    fi_t y;
    for (x = 0; x < n; ++x) {
        y.ui = ((uint64_t) 1023 << 52) | (x << fastlog_man_offset);
        fastlog_lookup[x] = log(y.f);
    }
}


void fastlog_free()
{
    free(fastlog_lookup); fastlog_lookup = NULL;
}


double log(double x)
{
    fi_t y;
    y.f = x;
    register const int exp  = (convert5((y.si >> 52))) - 1023;
    register const uint64_t man  = (y.ui & fastlog_man_mask) >> fastlog_man_offset;

    return M_LN2 * convert6(exp) + fastlog_lookup[man];
}
