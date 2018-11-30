
//# sudo gcc -c libapprox.c 
//# sudo ar crv libapprox.a libapprox.o 
// sudo libtool --mode=link gcc -export-dynamic -o libapprox.a libapprox.o

// __wrap_malloc  //gcc -ggdb -o test test.c -Wl,-wrap,malloc

#ifdef approx

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

#if __GNUC__
 #define ALIGN(x) x __attribute__((aligned(32)))
 #define SIMD_ALIGNED(var) var __attribute__((aligned(16)))
 #define ALIGN32_BEG __attribute__((aligned(32)))
#if __x86_64__ || __ppc64__
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif
#if defined(_MSC_VER)
  #define ALIGN(x) __declspec(align(32))
#endif

#define __SSE__ 1
#define __SSE2__ 1
#define __SSE2_MATH__ 1
#define __SSE_MATH__ 1

// FMA3 works greate with gcc Version >5
#define FP_FAST_FMA
#pragma STDC FP_CONTRACT ON

// 
#define FLT_MAX 3.402823e+38

// often used 
#define VECTORMATH_FORCE_INLINE inline 

//typedef __m256  v8sf; // vector of 8 float (avx)

//#define M_LOG2E 1.4426950408889634074 

typedef double double_t;

#include <stdint.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#define SFMT_N (19937 / 128 + 1)
#define SFMT_SR1 11
#define SFMT_SR2 1
#define SFMT_SL2 1
#define SFMT_SL1 18

union W128_T {
    uint32_t u[4];
    uint64_t u64[2];
    __m128i si;
};

typedef union W128_T w128_t;

struct SFMT_T {
    /** the 128-bit internal state array */
    w128_t state[SFMT_N];
    /** index counter to the 32-bit internal state array */
    int idx;
};

typedef struct SFMT_T sfmt_t;

#define vec_nmsub(a,b,c) _mm_sub_ps( c, _mm_mul_ps( a, b ) )
#define vec_sub(a,b) _mm_sub_ps( a, b )
#define vec_add(a,b) _mm_add_ps( a, b )
#define vec_mul(a,b) _mm_mul_ps( a, b )
#define vec_xor(a,b) _mm_xor_ps( a, b )
#define vec_and(a,b) _mm_and_ps( a, b )
#define vec_cmpeq(a,b) _mm_cmpeq_ps( a, b )
#define vec_cmpgt(a,b) _mm_cmpgt_ps( a, b )

#define vec_mergeh(a,b) _mm_unpacklo_ps( a, b )
#define vec_mergel(a,b) _mm_unpackhi_ps( a, b )

#define vec_andc(a,b) _mm_andnot_ps( b, a )

#define sqrtf4(x) _mm_sqrt_ps( x )
#define rsqrtf4(x) _mm_rsqrt_ps( x )
#define recipf4(x) _mm_rcp_ps( x )
#define negatef4(x) _mm_sub_ps( _mm_setzero_ps(), x )

#define vec_splat(x, e) _mm_shuffle_ps(x, x, _MM_SHUFFLE(e,e,e,e))

#define _mm_ror_ps(vec,i)	\
	(((i)%4) ? (_mm_shuffle_ps(vec,vec, _MM_SHUFFLE((unsigned char)(i+3)%4,(unsigned char)(i+2)%4,(unsigned char)(i+1)%4,(unsigned char)(i+0)%4))) : (vec))
#define _mm_rol_ps(vec,i)	\
	(((i)%4) ? (_mm_shuffle_ps(vec,vec, _MM_SHUFFLE((unsigned char)(7-i)%4,(unsigned char)(6-i)%4,(unsigned char)(5-i)%4,(unsigned char)(4-i)%4))) : (vec))

#define vec_sld(vec,vec2,x) _mm_ror_ps(vec, ((x)/4))

#define _mm_abs_ps(vec)		_mm_andnot_ps(_MASKSIGN_,vec)
#define _mm_neg_ps(vec)		_mm_xor_ps(_MASKSIGN_,vec)

#define vec_madd(a, b, c) _mm_add_ps(c, _mm_mul_ps(a, b) )

// optimization makros
//#define sin(x)*sin(x) ssin(x)
//#define cos(x)*cos(x) scos(x)
//#define sin(x)*cos(x) dsin(x)
//#define cos(x)*cos(x)-sin(x)*sin(x) dcos(x)
//#define sin(acos(x)) sh1(x)
//#define cos(asin(x)) sh1(x)
#undef sin
#undef cos
#undef tan
#undef sqrt
#undef pow
#undef sinh
#undef atan2
#undef atan
#undef abs
#undef asin


#ifndef ncp
#include <sse_mathfun.h>
//#include <avx_mathfun.h>
//#include <fma.h>
#include <FastMemcpy_Avx.h>
#include <nlopt.h>
#include <arrayfire.h>
//#define malloc(n) GC_malloc(n)
#define calloc(m,n) GC_malloc((m)*(n))
#define free(p) GC_free(p)
#define realloc(p,n) GC_realloc((p),(n))
#define check_leaks() GC_gcollect()
	#define CROSS_PRODUCT(a,b) _mm256_permute4x64_pd(\
		_mm256_sub_pd(\
			_mm256_mul_pd(a, _mm256_permute4x64_pd(b, _MM_SHUFFLE(3, 0, 2, 1))),\
			_mm256_mul_pd(b, _mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 0, 2, 1)))\
		), _MM_SHUFFLE(3, 0, 2, 1)\
	)
#endif
#include <stdint.h>
//#include <erfc.c>
#if defined (__has_include) && (__has_include(<x86intrin.h>))
#include <x86intrin.h>
#else
#error "Upgrade your Systen please thx..."
#endif

//#ifndef cos(x)
//#define cos(x) fastcos1(x)
//#endif
#define sin(x) sin1(x)
//#define sin(x) cos(x-M_PI_2)
/*
#ifndef cos(x)
#define cos(x) fastcos1(n)
#endif
#ifndef sin(x)
#define sin(x) fastsin1(x)
#endif
*/

//double erfc(double x) __attribute__ ((alias("erfca")));
double ceila(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float acosa(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float asina(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float sinha(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
double tanha(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
double loga(double)  __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float mina(float, float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float maxa(float, float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float rcpa(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//double sin(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//double cos(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//double tana(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
double cota(double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
double expa (double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
double powa (double,double) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float atana (float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float atan2a (float,float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float sqrta(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
float rsqrta(float ) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//float absa(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//float sincosa(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));

// decrease performance 
//float fastsin1(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));
//float fastcos1(float) __attribute__ ((hot)) __attribute__ ((__target__ ("default")));


// Avoiding AVX-SSE Transition Penalties  (dont use SSE and AVX at the same time)
// Converting costs ( -mavx -mavx2 )
#pragma GCC push_options
//#pragma GCC optimize ("-march=core-avx-i -lm -DHAVE_ACML -L/opt/sse -L/opt/OpenBLAS/lib/ -L/usr/local/lib/ -lmingw32 -mf16c -latlas -lfftw3 -llapack -lblas -lOpenCL -lcufft_static -lcublas_static -lclblast -lacml_mv -lamac64o -fopenmp -I/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/include -L/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64 -lmkl_core -lmkl_tbb_thread -lmkl_intel_ilp64 -lstdc++ -lpthread -O1 -funroll-loops -flto -ftree-vectorize -funsafe-math-optimizations -mveclibabi=svml -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mrtm -fipa-cp -ftree-loop-distribution -fprefetch-loop-arrays -freorder-blocks-and-partition -fif-conversion -ftree-loop-if-convert -freorder-blocks -ftree-loop-if-convert -ftree-vectorize -fweb -ftree-loop-if-convert -fsplit-wide-types -ftree-slp-vectorize -ftree-dse -fgcse-las -fsched-dep-count-heuristic -fno-tree-slsr -fsched-spec-load -fconserve-stack -fstrict-aliasing -free -ftree-vrp -fthread-jumps --param=max-reload-search-insns=356 --param=max-cselib-memory-locations=1202 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
#pragma GCC optimize ("-s -static -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mfma -funsafe-math-optimizations -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) -mpreferred-stack-boundary=3 -march=native -mtune=native -O1 -funroll-loops -ftree-vectorize -mveclibabi=svml -fprefetch-loop-arrays  -ftree-parallelize-loops=4 -mrtm -fipa-cp -ftree-loop-distribution -fprefetch-loop-arrays -freorder-blocks-and-partition -fif-conversion -ftree-loop-if-convert -freorder-blocks -ftree-loop-if-convert -ftree-vectorize -fweb  -ftree-loop-if-convert -fsplit-wide-types -ftree-slp-vectorize -ftree-dse -fgcse-las -fsched-dep-count-heuristic -fno-tree-slsr -fsched-spec-load -fconserve-stack -fstrict-aliasing -free -ftree-vrp -fthread-jumps -maccumulate-outgoing-args -msseregparm -minline-all-stringops -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc --param=max-reload-search-insns=356 --param=max-cselib-memory-locations=2000 --param=max-sched-ready-insns=1000 --param=max-crossjump-edges=30 --param=max-delay-slot-insn-search=137 -fno-guess-branch-probability -fno-if-conversion -fno-ivopts -fno-schedule-insns -fsingle-precision-constant --param=max-unswitch-insns=5 --param=l1-cache-size=$((128*1/100)) --param=l2-cache-size=512")

static const double Zero[] = {0.0, -0.0,};
#define packed_double(x) {(x), (x)}

#define M_LOG2E 1.4426950408889634074
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144  
#define I_PI	    0.31830988618379067153776752674502872
#define Eul	    2.7182818284590452353602874

#define fsl_PI 3.1415926535897932384626433f
#define fsl_HALF_PI 1.57079632679


#define     EQN_EPS     1e-9
#define	    IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)

#ifdef NOCBRT
#define     cbrt(x)     ((x) > 0.0 ? pow((double)(x), 1.0/3.0) : \
                          ((x) < 0.0 ? -pow((double)-(x), 1.0/3.0) : 0.0))
#endif

static const uint32_t
B1 = 715094163, /* B1 = (1023-1023/3-0.03306235651)*2**20 */
B2 = 696219795; /* B2 = (1023-1023/3-54/3-0.03306235651)*2**20 */

/* |1/cbrt(x) - p(x)| < 2**-23.5 (~[-7.93e-8, 7.929e-8]). */
static const double
P0 =  1.87595182427177009643,  /* 0x3ffe03e6, 0x0f61e692 */
P1 = -1.88497979543377169875,  /* 0xbffe28e0, 0x92f02420 */
P2 =  1.621429720105354466140, /* 0x3ff9f160, 0x4a49d6c2 */
P3 = -0.758397934778766047437, /* 0xbfe844cb, 0xbee751d9 */
P4 =  0.145996192886612446982; /* 0x3fc2b000, 0xd4e4edd7 */


const float __log10f_rng =  0.3010299957f;

#define ALIGNED __attribute__((aligned(16)))

#define packed_double(x) {(x), (x)}

double coef3[2] ALIGNED = packed_double(1.0/6);
double coef5[2] ALIGNED = packed_double(1.0/120);
double coef7[2] ALIGNED = packed_double(1.0/5040);
double coef9[2] ALIGNED = packed_double(1.0/362880);

//float to int
#define FLOAT_FTOI_MAGIC_NUM (float)(3<<21)
#define IT_FTOI_MAGIC_NUM (0x4ac00000)

// cache last sin(x)
// n =  x , (n+1) = sin(x)
#define MAX_CIRCLE_ANGLE      512
#define HALF_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/2)
#define QUARTER_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE/4)
#define MASK_MAX_CIRCLE_ANGLE (MAX_CIRCLE_ANGLE - 1)
float fast_cossin_table[MAX_CIRCLE_ANGLE];

unsigned short sincache[2] = { 0, 0 };
unsigned short coscache[2] = { 0, 0 };
float expcache[33] = { };
unsigned short logcache[2] = { 0, 0 };
unsigned short SinTable[256];

//#pragma GCC push_options
//#pragma GCC optimize ("-s -static -funsafe-math-optimizations -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) -mpreferred-stack-boundary=3 -march=native -mtune=native -O1 -funroll-loops -ftree-vectorize -mveclibabi=svml -fprefetch-loop-arrays  -ftree-parallelize-loops=4 -mrtm -fipa-cp -ftree-loop-distribution -fprefetch-loop-arrays -freorder-blocks-and-partition -fif-conversion -ftree-loop-if-convert -freorder-blocks -ftree-loop-if-convert -ftree-vectorize -fweb  -ftree-loop-if-convert -fsplit-wide-types -ftree-slp-vectorize -ftree-dse -fgcse-las -fsched-dep-count-heuristic -fno-tree-slsr -fsched-spec-load -fconserve-stack -fstrict-aliasing -free -ftree-vrp -fthread-jumps -maccumulate-outgoing-args -msseregparm -minline-all-stringops -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc --param=max-reload-search-insns=356 --param=max-cselib-memory-locations=2000 --param=max-sched-ready-insns=1000 --param=max-crossjump-edges=30 --param=max-delay-slot-insn-search=137 -fno-guess-branch-probability -fno-if-conversion -fno-ivopts -fno-schedule-insns -fsingle-precision-constant --param=max-unswitch-insns=5 --param=l1-cache-size=$((128*1/100)) --param=l2-cache-size=512")

// convert float to int (int)
int convert(float x)
{
    return _mm_cvtt_ss2si(_mm_load_ss(&x)); // extra 't' means truncate
}

// int to float (float)
int convert2(float f)
{
	f += FLOAT_FTOI_MAGIC_NUM;
	return (*((int*)&f) - IT_FTOI_MAGIC_NUM)>>1;
}

// convert int to float
float convert3(int x)
{
	//__m128i  = {0,0,0,0};
	//_mm256_cvtps_epi32
	//return _mm_cvtps_epi32(_mm_set1_ps(&x));
	//return _mm_cvt_ss2si(_mm_load_ss(&f)); 
}

float inv_fast(float x) {
	union { float f; int i; } v;
	float w, sx;
	int m;

	sx = (x < 0) ? -1:1;
	x = sx * x;

	v.i = convert((0x7EF127EA - *(uint32_t *)&x));
	w = x * v.f;

	// Efficient Iterative Approximation Improvement in horner polynomial form.
	v.f = v.f * (2 - w);     // Single iteration, Err = -3.36e-3 * 2^(-flr(log2(x)))
	// v.f = v.f * ( 4 + w * (-6 + w * (4 - w)));  // Second iteration, Err = -1.13e-5 * 2^(-flr(log2(x)))
	// v.f = v.f * (8 + w * (-28 + w * (56 + w * (-70 + w *(56 + w * (-28 + w * (8 - w)))))));  // Third Iteration, Err = +-6.8e-8 *  2^(-flr(log2$
	return v.f * sx;
}


float cos(float n)
{
   if(unlikely(fast_cossin_table[45]==0))
   {
   InitSinTable();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&fast_cossin_table,1,1);
   d = fast_cossin_table[(i + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
	return fast_cossin_table[((-i) + QUARTER_MAX_CIRCLE_ANGLE)&MASK_MAX_CIRCLE_ANGLE];
   }
   return d;
}

inline float sinwrapper(float n)
{
	return cos(n-M_PI_2);
}


static float sin1(float n)
{
   if(unlikely(fast_cossin_table[45]==0))
   {
   InitSinTable();
   }
   float d;
   float f = n * HALF_MAX_CIRCLE_ANGLE * I_PI;
   int i =  _mm_cvtt_ss2si(_mm_load_ss(&f));
   const int c=0x80000000;
   __builtin_prefetch(&fast_cossin_table,1,1);
   d = fast_cossin_table[i&MASK_MAX_CIRCLE_ANGLE];
   if (unlikely(i&c))
   {
      return fast_cossin_table[(-((-i)&MASK_MAX_CIRCLE_ANGLE)) + MAX_CIRCLE_ANGLE];
   }
   return d;


}

//#pragma GCC pop_options

inline __m256 sftoavxa(float a[7])
{
	__m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
	return f;
}

inline __m256 s2ftoavxa(float a[7],float b[7])
{
        __m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        __m256 g = _mm256_set_ps(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]); 
}

//# unsigned short 
// short 16 bit
// int   32 bit 
inline __m256 hftoavx(unsigned short c[15])
{
	int i,a[7];
	for(i=0;i<16;i=i+2){
	a[i]=c[i];
	a[i]=c[i]<<8;
	a[i]=a[i]|c[i+1];
	}
        //__m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        //return f;
}

inline float ftoaadd(float a[7], float b[7],float *d)
{
        //float d[8];  // call z[8];ftoaadd(g,f,z);
        __m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        __m256 g = _mm256_set_ps(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
        __m256 c = _mm256_add_ps(f, g);
        _mm256_storeu_ps(d, c);
        //return d;
}

inline float ftomul(float a[7], float b[7],float *d)
{
        __m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        __m256 g = _mm256_set_ps(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
        __m256 c = _mm256_mul_ps(f, g);
        _mm256_storeu_ps(d, c);
}

inline float ftosub(float a[7], float b[7],float *d)
{
        __m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        __m256 g = _mm256_set_ps(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
        __m256 c = _mm256_sub_ps(f, g);
        _mm256_storeu_ps(d, c);
}

inline __m256 mul_addv(float a[7], float b[7],float c[7],float *d) 
{
        __m256 f = _mm256_set_ps(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
        __m256 g = _mm256_set_ps(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]);
        __m256 h = _mm256_set_ps(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]);
        //__m256 c = _mm256_add_ps(f, g);
        _mm256_storeu_ps(d, _mm256_fmadd_ps(f,g,h));

	//return a*b + c;
	//__m256 q1 = __mm256_add_ps(_mm256_mul_ps(a, b), c);
	//return _mm256_fmadd_ps(a,b,c);
}

__m256 cfloattoavx(float x) 
{
	// Load the float array into an avx vector
	//__m256 vect = _mm256_load_ps(a);
	__m256 v = _mm256_set1_ps(x);
	return v;
}

// round float to int 
/*int rftoint(float f)
{
__m128 v =  _mm_cvtt_ss2si(cfto4f((f));
}*/

float cavxtofloat(float x)
{
	float d[8];
	__m256 y;
	y = cfloattoavx(x);
	_mm256_storeu_ps(d, y);
	// *(__m256)(v) = _mm256_dp_ps(a, b, 0xff);
	return d[0];
}

unsigned short f32tof16(float x)
{
    //float x;
    unsigned short f16;
    f16 = _cvtss_sh(x, 0);
    return f16;
}

float f16tof32(unsigned short x)
{
	float f32;
	f32 = _cvtsh_ss(x);
	return f32;
}

//  sin(x +π/2) = cos(x)
//
#undef sin(x)
#pragma omp parallel
#pragma omp for
void InitSinTable()
{
	int i;
	for (i = 0 ; i < MAX_CIRCLE_ANGLE ; i++)
	{
		fast_cossin_table[i] = (float)sin((double)i * M_PI / HALF_MAX_CIRCLE_ANGLE);
	}
}
#define sin(x) sin1(x)

inline double log2(double n)
{
    return log(n) * M_LOG2E;
}

float reciprocal( float x )
{
    union {
        float dbl;
        unsigned uint;
    } u;
    u.dbl = x;
    u.uint = ( 0xbe6eb3beU - u.uint ) >> (unsigned char)1;
                                    // pow( x, -0.5 )
    u.dbl *= u.dbl;                 // pow( pow(x,-0.5), 2 ) = pow( x, -1 ) = 1.0 / x
    return u.dbl;
}

float mmul(const float * a, const float * b, float * r)
{
	__m128 a_line, b_line, r_line;
	int i, j;
	#pragma omp parallel
	#pragma omp for
	for (i=0; i<16; i+=4) {
    // unroll the first step of the loop to avoid having to initialize r_line to zero
		a_line = _mm_load_ps(a);         // a_line = vec4(column(a, 0))
		b_line = _mm_set1_ps(b[i]);      // b_line = vec4(b[i][0])
		r_line = _mm_mul_ps(a_line, b_line); // r_line = a_line * b_line
		for (j=1; j<4; j++) {
			a_line = _mm_load_ps(&a[j*4]); // a_line = vec4(column(a, j))
			b_line = _mm_set1_ps(b[i+j]);  // b_line = vec4(b[i][j])
                                     // r_line += a_line * b_line
			r_line = _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line);
		}
		_mm_store_ps(&r[i], r_line);     // r[i] = r_line
	}
}


// Instruction  	Operands 	Ops Latency 	Reciprocal throughput
// AND, OR, XOR 	r,r 		1   1		1/3 ALU
// AND, OR, XOR 	r,m 		1   1 		1/2 ALU, AGU
// AND, OR, XOR 	m,r 		1   7 		2,5 ALU, AGU


void swap(int *x, int *y) 
{
	int *a[2];
	a[0]=x;
	a[1]=y;
	__builtin_prefetch(&a,1,1); 
	if (a[0] == a[1]) 
	{
		return;
	}
	else
	{
	*a[0] = *a[0] + *a[1];
	*a[1] = *a[0] - *a[1];
	*a[0] = *a[0] - *a[1];
	return(*a[0],*a[1]);	
	}
}

const float __sincosf_rng[2] = {
	2.0 / M_PI,
	M_PI / 2.0
};

const float __sincosf_lut[8] = {
	-0.00018365f,	//p7
	-0.00018365f,	//p7
	+0.00830636f,	//p5
	+0.00830636f,	//p5
	-0.16664831f,	//p3
	-0.16664831f,	//p3
	+0.99999661f,	//p1
	+0.99999661f,	//p1
};

void sincosa( float x, float r[2])
{
	union {
		float 	f;
		int 	i;
	} ax, bx;
	
	float y;
	float a, b, c, d, xx, yy;
	int m, n, o, p;
	
	y = x + __sincosf_rng[1];
	ax.f = fabsf(x);
	bx.f = fabsf(y);
	
	//Range Reduction:
	m = convert((ax.f * __sincosf_rng[0]));	
	o = convert((bx.f * __sincosf_rng[0]));	
	ax.f = ax.f - ((convert(m)) * __sincosf_rng[1]);
	bx.f = bx.f - ((convert(o)) * __sincosf_rng[1]);
	
	//Test Quadrant
	n = m & 1;
	p = o & 1;
	ax.f = ax.f - n * __sincosf_rng[1];	
	bx.f = bx.f - p * __sincosf_rng[1];	
	m = m >> 1;
	o = o >> 1;
	n = n ^ m;
	p = p ^ o;
	m = (x < 0.0);
	o = (y < 0.0);
	n = n ^ m;	
	p = p ^ o;	
	n = n << 31;
	p = p << 31;
	ax.i = ax.i ^ n; 
	bx.i = bx.i ^ p; 

	//Taylor Polynomial
	xx = ax.f * ax.f;	
	yy = bx.f * bx.f;
	r[0] = __sincosf_lut[0];
	r[1] = __sincosf_lut[1];
	r[0] = r[0] * xx + __sincosf_lut[2];
	r[1] = r[1] * yy + __sincosf_lut[3];
	r[0] = r[0] * xx + __sincosf_lut[4];
	r[1] = r[1] * yy + __sincosf_lut[5];
	r[0] = r[0] * xx + __sincosf_lut[6];
	r[1] = r[1] * yy + __sincosf_lut[7];
	r[0] = r[0] * ax.f;
	r[1] = r[1] * bx.f;

}

float fmod(float x, float y)
{
	int n;
	union {
		float f;
		int   i;
	} yinv;
	float a;
	
	//fast reciporical approximation (4x Newton)
	yinv.f = y;
	n = 0x3F800000 - (yinv.i & 0x7F800000);
	yinv.i = yinv.i + n;
	yinv.f = 1.41176471f - 0.47058824f * yinv.f;
	yinv.i = yinv.i + n;
	a = 2.0 - yinv.f * y;
	yinv.f = yinv.f * a;	
	a = 2.0 - yinv.f * y;
	yinv.f = yinv.f * a;
	a = 2.0 - yinv.f * y;
	yinv.f = yinv.f * a;
	a = 2.0 - yinv.f * y;
	yinv.f = yinv.f * a;
	
	n = convert2((x * yinv.f));
	x = x - (convert(n)) * y;
	return x;
}

float invsqrtfa(float x)
{

	float b, c;
	union {
		float 	f;
		int 	i;
	} a;
	
	//fast invsqrt approx
	a.f = x;
	a.i = 0x5F3759DF - (a.i >> 1);		//VRSQRTE
	c = x * a.f;
	b = (3.0f - c * a.f) * 0.5;		//VRSQRTS
	a.f = a.f * b;		
	c = x * a.f;
	b = (3.0f - c * a.f) * 0.5;
    a.f = a.f * b;	

	return a.f;
}

const float __log10f_lut[8] = {
	-0.99697286229624, 		//p0
	-1.07301643912502, 		//p4
	-2.46980061535534, 		//p2
	-0.07176870463131, 		//p6
	2.247870219989470, 		//p1
	0.366547581117400, 		//p5
	1.991005185100089, 		//p3
	0.006135635201050,		//p7
};

float log10(float x)
{
	float a, b, c, d, xx;
	int m;
	
	union {
		float   f;
		int 	i;
	} r;
	
	//extract exponent
	r.f = x;
	m = (r.i >> 23);
	m = m - 127;
	r.i = r.i - (m << 23);
		
	//Taylor Polynomial (Estrins)
	xx = r.f * r.f;
	a = (__log10f_lut[4] * r.f) + (__log10f_lut[0]);
	b = (__log10f_lut[6] * r.f) + (__log10f_lut[2]);
	c = (__log10f_lut[5] * r.f) + (__log10f_lut[1]);
	d = (__log10f_lut[7] * r.f) + (__log10f_lut[3]);
	a = a + b * xx;
	c = c + d * xx;
	xx = xx * xx;
	r.f = a + c * xx;

	//add exponent
	r.f = r.f + (convert2(m)) * __log10f_rng;

	return r.f;
}

void sin_sse(const double arg1, const double arg2, double* sin_x1, double* sin_x2) {

	__asm__ volatile (
		"movlpd %0, %%xmm7			    \n"
		"movhpd %1, %%xmm7			    \n" // xmm7 = [arg2, arg1]
		"movapd %%xmm7, %%xmm6			\n" 

		"movapd %%xmm7, %%xmm0			\n"
		"mulpd  %%xmm6, %%xmm6			\n" // xmm6 = x^2 [x2]
		"mulpd  %%xmm6, %%xmm0			\n" // xmm0 = x^3 [x3]

		"movapd %%xmm6, %%xmm1			\n"
		"movapd %%xmm6, %%xmm2			\n"
		"movapd %%xmm6, %%xmm3			\n"
		"mulpd  %%xmm0, %%xmm1			\n" // xmm1 = x^5 [x5]
		"mulpd  %%xmm1, %%xmm2			\n" // xmm2 = x^7 [x7]
		"mulpd  %%xmm2, %%xmm3			\n" // xmm3 = x^9 [x9]

		"mulpd   coef3, %%xmm0			\n" // xmm0 = x^3/6
		"mulpd   coef5, %%xmm1			\n" // xmm1 = x^5/120
		"mulpd   coef7, %%xmm2			\n" // xmm2 = x^7/5040
		"mulpd   coef9, %%xmm3			\n" // xmm3 = x^7/362800

		// final sum
		"subpd  %%xmm0, %%xmm7			\n"
		"addpd  %%xmm1, %%xmm7			\n"
		"subpd  %%xmm2, %%xmm7			\n"
		"addpd  %%xmm3, %%xmm7			\n"

		"movhpd %%xmm7, %2				\n"
		"movlpd %%xmm7, %3				\n"
		: /* no output */
		: "m" (arg1),
		  "m" (arg2),
		  "m" (*sin_x1),
		  "m" (*sin_x2)
		: /* nothing is modified */
	);
}

double sina(double x) 
{
	double result = -1.0;
	sin_sse(x, x, &result, &result);
	return result;
}

double cosaa(double x)
{
	return sina(x + M_PI_2);
}


double ceila(double x)
{
        int i = convert(x);
        return (x > (i)) ? i+1 : i;
}

static float floora(float x)
{
  __m128 f = _mm_set_ss(x);
  __m128 one = _mm_set_ss(1.0f);
  __m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(f));
  __m128 r = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(f, t), one));
  return _mm_cvtss_f32(r);
}


//   const int n = 4; /* small enough to fit in cache */
//   int a[n] __attribute__ ((aligned (16))); /* align the array in memory by 16 bytes */
//   for( int i = 0; i < n; i++ ) a[i] = rand( );
int sum( int n, int *a )
{
    __m128i sum = _mm_setzero_si128( );
    int val[4];
    #pragma omp parallel
    #pragma omp for
    for ( int i = 0; i < n/4*4; i += 4 ) {
        __m128i vect = _mm_loadu_si128( (__m128i*) (a+i) );
        sum = _mm_add_epi32( vect, sum );
    }
    _mm_storeu_si128( (__m128i*) val, sum);
    int res = val[0] + val[1] + val[2] + val[3];
    #pragma omp parallel
    #pragma omp for
    for( int i = n/4*4; i < n; i += 1 )
        res += a[i];
    return res;
}

float rcpNewtonRaphsona(float inX, float inRcpX)
{
	return inRcpX * (-inRcpX * inX + 2.0f);
}

float acosa(float x) {
   return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966;
}

inline float asina(float inX)
{
	float x = inX;
	return fsl_HALF_PI - acosa(x);
}

inline float sinha(float x)
{
	return 0.5 * (expa(x)-expa(-x));
}

inline double tanha(double x)
{
	return -1.0f + 2.0f / (1.0f + expa (-2.0f * x));
}

// half float decrease values that the probability of repeating increase
//

inline double expa(double a)
{	
	if(a<0.2)
	{
	return (1+a);
	}
	const int i[2] = { 1512775,1072632447 };
	__builtin_prefetch(&i,1,1);
	union { double d; int x[2]; } u;
	u.x[1] = convert((i[0] * a + i[1]));
	u.x[0] = 0;
	//printf("%f",f16tof32(expcache[0]));
	return u.d;
}

double loga(double a) 
{
	double y=0;
	union { double d; long long x; } u = { a };
	y = (u.x - 4607182418800017409) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
	return y;
}


inline double powa(double a, double b) 
{
	union { double d; int x[2]; } u = { a };
	u.x[1] = convert((b * (u.x[1] - 1072632447) + 1072632447));
	u.x[0] = 0;
	return u.d;
}

float rsqrta(float x)
{
return powa(x, -0.5);
}

// performance by div is probably higher by double
// add sub & mul are inverse 
float sqrta(float x)
{
return inv_fast(rsqrta(x));
}

float atana(float inX)
{
	const float i[3] = { -0.1784f,0.0663f,1.0301f };
	float  x = inX;
	return x*(i[0] * abs(x) - i[1] * x * x + i[2]);
}


float atan2a( float y, float x )
{
	const float i[5] = { 3.14159265f,1.5707963f,0.0f,1.0f,0.28f };
	if ( x == i[2] )
	{
		if ( y > i[2] ) return i[1];
		if ( y == i[2] ) return i[2];
		return -i[1];
	}
	float atan;
	float z = y/x;
	if ( fabs( z ) < i[3] )
	{
		atan = z/(i[3] + i[4]*z*z);
		if ( x < i[2] )
		{
			if ( y < i[2] ) return atan - i[0];
			return atan + i[0];
		}
	}
	else
	{
		atan = i[1] - z/(z*z + i[4]);
		if ( y < i[2] ) return atan - i[0];
	}
	return atan;
}

float signnza(float op)
{
	__m128 v = _mm_set_ss(op);
	__m128 s = _mm_or_ps(_mm_and_ps(v, _mm_set_ss(-0.0f)), _mm_set_ss(1.0f));
	return _mm_cvtss_f32(s);
}

float signa(float op)
{
	__m128 v = _mm_set_ss(op);
	__m128 s = _mm_or_ps(_mm_and_ps(v, _mm_set_ss(-0.0f)), _mm_set_ss(1.0f));
	__m128 nz = _mm_cmpneq_ps(v, _mm_setzero_ps());
	__m128 s3 = _mm_and_ps(s, nz);
	return _mm_cvtss_f32(s3);
}

float sqrtsse(float op) 
{
return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(op)));
}

float mina(float v0, float v1)
{
	return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(v0), _mm_set_ss(v1)));
}

float maxa(float v0,float v1)
{
	return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(v0), _mm_set_ss(v1)));
}

float rcpa(float op)   
{ 
return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(op))); 
}

inline float FastClampInfinity( float x )
{
	return ( x > FLT_MAX ? FLT_MAX : x );
}


double ssin(double x)
{
double b=0;
b=(1-cos(2*x))/2;
return (b);
}

double scos(double x)
{
double b=0;
b=(1+cos(2*x))/2;
return (b);
}

double dsin(double x)
{
return sina(2*x);
}

double dcos(double x)
{
return cos(2*x);
}

double sh1(double x)
{
return (sqrt(1-(x*x)));
}

// sin(0.3)=0.29552
/*inline double sin2(double x)
{
  if(x<0.3)
  {
  return(x);
  }
  if(0==fmod(x,90))
  {
  return(0);
  }
  double i[5] = {0.40528473456935108577551785283891,0.2248391013559941,1};
  i[3] = x;
  __builtin_prefetch(&i,1,1); 
  double y = i[0]* i[3] * ( M_PI - i[3] );
  y = y* ( (i[2]-i[1])  + y * i[1] );
  return y;
}*/

inline double cosa(double x)
{
  if(0==fmod(x,90))
  {
  return(1);
  }
  double input = x;
  return sin(x+M_PI_2);
}

double tan(double x)
{
	double input = x;
	return (sin(x)*inv_fast(cos(x)));
}

double cota(double x)
{
  double input = x;
  return (cos(x)/sin(x));
}


float absa(float x)
{
int casted = *(int*) &x;
casted &= 0x7FFFFFFF;
return *(float*)&casted;
}

float Factorial(float value)
{
float sum=0,value1=0;
value1=(value/Eul);
sum=sqrta(2*value*M_PI)*powa(value1,value);
return sum;
}

int Factorial2(int facno)
{
    int temno = 1;
    #pragma omp parallel
    #pragma omp for
    for (int i = 1; i <= facno; i++)
    {
        temno = temno * i;
    }
    return temno;
}

float arisum(float value)
{
float sum=0;
sum=(0.5*value*(value+1));
return sum;
}

float arisumqua(float value)
{
float sum=0;
sum=((value*(value+1)*(2*value+1))/6);
return sum;
}

float arisumkub(float value)
{
float sum=0;
sum=(value*value*((value+1)*(value+1))*0.25);
return sum;
}


double cbrt2(double x)
{
	union {double f; uint64_t i;} u = {x};
	double_t r,s,t,w;
	uint32_t hx = u.i>>32 & 0x7fffffff;

	if (hx >= 0x7ff00000)  /* cbrt(NaN,INF) is itself */
		return x+x;

	if (hx < 0x00100000) { /* zero or subnormal? */
		u.f = x*0x1p54;
		hx = u.i>>32 & 0x7fffffff;
		if (hx == 0)
			return x;  /* cbrt(0) is itself */
		hx = hx/3 + B2;
	} else
		hx = hx/3 + B1;
	u.i &= 1ULL<<63;
	u.i |= (uint64_t)hx << 32;
	t = u.f;

	r = (t*t)*(t/x);
	t = t*((P0+r*(P1+r*P2))+((r*r)*r)*(P3+r*P4));

	u.f = t;
	u.i = (u.i + 0x80000000) & 0xffffffffc0000000ULL;
	t = u.f;

	/* one step Newton iteration to 53 bits with error < 0.667 ulps */
	s = t*t;         /* t*t is exact */
	r = x/s;         /* error <= 0.5 ulps; |r| < |t| */
	w = t+t;         /* t+t is exact */
	r = (r-t)/(w+r); /* r-t is exact; w+r ~= 3*t */
	t = t+t*r;       /* error <= 0.5 + 0.5/3 + epsilon */
	return t;
}

int VectorsEqual( const float *v1, const float *v2 )
{
	return ( ( v1[0] == v2[0] ) && ( v1[1] == v2[1] ) && ( v1[2] == v2[2] ) );
}

int VectorCompare (const float *v1, const float *v2)
{
	int i;
	#pragma omp parallel
	#pragma omp for
	for (i=0 ; i<3 ; i++){
		if (v1[i] != v2[i]){
			return 0;
		}}			
	return 1;
}

void CrossProduct (const float* v1, const float* v2, float* cross)
{
	cross[0] = v1[1]*v2[2] - v1[2]*v2[1];
	cross[1] = v1[2]*v2[0] - v1[0]*v2[2];
	cross[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

int GreatestCommonDivisor (int i1, int i2)
{
	if (i1 > i2)
	{
		if (i2 == 0)
			return (i1);
		return GreatestCommonDivisor (i2, i1 % i2);
	}
	else
	{
		if (i1 == 0)
			return (i2);
		return GreatestCommonDivisor (i1, i2 % i1);
	}
}

double distance_2_points(float x1, float y1, float x2, float y2) {
    float distance;
    float distance_x = x1 - x2;
    float distance_y = y1 - y2;
    distance = sqrt((distance_x * distance_x) + (distance_y * distance_y));
    return distance;
}

static VECTORMATH_FORCE_INLINE __m128 vec_sel(__m128 a, __m128 b, __m128 mask)
{
	return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static VECTORMATH_FORCE_INLINE __m128 toM128(unsigned int x)
	{
		return _mm_set1_ps( *(float *)&x );
	}


static VECTORMATH_FORCE_INLINE __m128 fabsf4(__m128 x)
{
    return _mm_and_ps( x, toM128( 0x7fffffff ) );
}

/*
int iszero(double x)
{
    union
    {
        float f;
        int i;
    } u;
    u.f = (float)x;
    u.i &= 0x7FFFFFFF;
    return -((u.i >> 31) ^ (u.i - 1) >> 31);
}*/

#if defined(ENVIRONMENT32)
int iszero(int x) {   return -(x >> 31 ^ (x - 1) >> 31); }
#endif
#if defined(ENVIRONMENT64)
int iszero(int x) { return -(x >> 63 ^ (x - 1) >> 63); }
#endif
float sqrt_tpl(float op) { __m128 s = _mm_sqrt_ss(_mm_set_ss(op));    float r; _mm_store_ss(&r, s);   return r; }
float fres(const float _a) { return 1.f / _a; }
float fsel(const float _a, const float _b, const float _c) { return (_a < 0.0f) ? _c : _b; }
float fmod_tpl(float x, float y) {return (float)fmod(x, y);}
int int_round(double f)  { return f < 0.0 ? convert((f - 0.5)) : convert2((f + 0.5)); }
int pos_round(double f)  { return convert((f + 0.5)); }


// Short Approximation Taylor 
// exp(-x)
float approxExp(float x) { return fres(1.f + x); }
// 1/(exp)
float approxInvExp(float x) { return (1 - x); }
// log(x+1)
float approxLog(float x) { return (x-(x*x)/2); }
// 1/log(x+1)
float approxInvLog(float x) { return (1/x + 0.5); }
// sin(x)
float approxSin(float x) { return (x-(x*x*x)/6); }
//  1/sin(x)
float approxInvSin(float x) { return (1/x+x/6); }
// sin(x)^2
float approxQuadSin(float x) { return (x*x-(x*x*x*x)/3); }
// x^x
float approxX(float x) { return (1+x*log(x)); }
// x^(-x)
float approxInvX(float x) { return (1-x*log(x)); }

/*
inline void vec3_add(float r, float a, float b) {
    int i;
    for (i = 0; i < 3; ++i) r[i] = a[i] + b[i];
}
inline void vec3_sub(float r, float a, float b) {
    int i;
    for (i = 0; i < 3; ++i) r[i] = a[i] - b[i];
}
inline void vec3_scale(float r, float v, float s) {
    int i;
    for (i = 0; i < 3; ++i) r[i] = v[i] * s;
}
*/


//#if defined(LIBM_ALIAS)
/* include aliases to the equivalent libm functions for use with LD_PRELOAD. */ 
//double log(double x) __attribute__ ((alias("loga"))); 
//double exp(double x) __attribute__ ((alias("fm_exp"))); 
//double exp10(double x) __attribute__ ((alias("fm_exp10"))); 
//float exp2f(float x) __attribute__ ((alias("fm_exp2f"))); 
//float expf(float x) __attribute__ ((alias("fm_expf"))); 
//float exp10f(float x) __attribute__ ((alias("fm_exp10f")));


//double cos(double x) __attribute__ ((alias("fastcos1")));
//double sin(double x) __attribute__ ((alias("fastsin1")));
//double fmod(double x) __attribute__ ((alias("fmoda")));
double log(double x) __attribute__ ((alias("loga")));
//double tan(double x) __attribute__ ((alias("tana")));
double exp(double x) __attribute__ ((alias("expa")));
double ceil(double x) __attribute__ ((alias("ceila")));
double floor(double x) __attribute__ ((alias("floora")));
float asin(float x) __attribute__ ((alias("asina")));
float sinh(float x) __attribute__ ((alias("sinha")));
double tanh(double x) __attribute__ ((alias("tanha")));
float max(float x) __attribute__ ((alias("maxa")));
float min(float x) __attribute__ ((alias("mina")));
float rcp(float x) __attribute__ ((alias("rcpa")));
float cot(float x) __attribute__ ((alias("cota")));
float pow(float x) __attribute__ ((alias("powa")));
float atan(float x) __attribute__ ((alias("atana")));
float atan2(float x) __attribute__ ((alias("atan2a")));
float sqrt(float x) __attribute__ ((alias("sqrta")));
float abs(float x) __attribute__ ((alias("absa")));
float sincos(float x) __attribute__ ((alias("sincosa")));


#pragma GCC pop_options
#endif





