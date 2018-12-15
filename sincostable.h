#include <stdio.h>
#include <stdlib.h>
//#include <math.h>

#include <emmintrin.h>
#include <xmmintrin.h>
#include <x86intrin.h>

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)


#define M_LOG2E 1.4426950408889634074
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144  
#define I_PI	    0.31830988618379067153776752674502872
#define Eul	    2.7182818284590452353602874

#define MAX_CIRCLE_ANGLE      512
#define HALF_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/2)
#define QUARTER_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/4)
#define MASK_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE - 1)



extern double sin ( double );

#pragma pack(push,1)
typedef struct triglist{
        float fast_cossin_table[MAX_CIRCLE_ANGLE] __attribute__ ((aligned (16)));
};
#pragma pack(pop)
struct triglist lib;


void InitSinTable()
{
        unsigned int i;
        #pragma omp parallel
        #pragma omp for
        for (i = 0 ; i < MAX_CIRCLE_ANGLE ; i++)
        {
                lib.fast_cossin_table[i] = (float)sin((double)i * M_PI / HALF_MAX_CIRCLE_ANGLE);
        }
}

float cos1(float n)
{
   if(unlikely(lib.fast_cossin_table[45]==0))
   {
   InitSinTable();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&lib.fast_cossin_table[0],1,1);
   d = lib.fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
        return lib.fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   }
   return d;
}

float sin1(float n)
{
   if(unlikely(lib.fast_cossin_table[45]==0))
   {
   InitSinTable();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&lib.fast_cossin_table[0],1,1);
   d = lib.fast_cossin_table[i&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
      return lib.fast_cossin_table[(-((-i)&MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE];
   }
   return d;
}

