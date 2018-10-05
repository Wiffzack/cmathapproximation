//# sudo gcc -c libapprox.c
//# sudo ar crv libapprox.a libapprox.o
// sudo libtool --mode=link  gcc -export-dynamic -o libapprox.a libapprox.o

// __wrap_malloc  //gcc -ggdb -o test test.c -Wl,-wrap,malloc

#ifdef approx

#define __SSE__ 1
#define __SSE2__ 1
#define __SSE2_MATH__ 1
#define __SSE_MATH__ 1

//#define M_LOG2E 1.4426950408889634074 

typedef double double_t;

// optimization makros
//#define sin(x)*sin(x) ssin(x)
//#define cos(x)*cos(x) scos(x)
//#define sin(x)*cos(x) dsin(x)
//#define cos(x)*cos(x)-sin(x)*sin(x) dcos(x)
//#define sin(acos(x)) sh1(x)
//#define cos(asin(x)) sh1(x)

#ifndef ncp
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

#ifndef cos(x)
#define cosa(x) cos(x)
#endif
#ifndef sin(x)
#define sina(x) sin(x)
#endif
#ifndef tan(x)
#define tana(a) tan(x)
#endif
//#ifndef exp(x)
//#define exp(x) expa(x)
//#endif
//#ifndef log(x)
//#define log(x) loga(x)
//#endif
//#ifndef sqrt(x)
//#define sqrt(x) sqrta(x)
//#endif
//#ifndef pow(x,y)
//#define pow(x,y) powa(x,y)
//#endif
//#ifndef atan(x)
//#define atan(x) atana(x)
//#endif
//#ifndef atan2(y,x)
//#define atan2(y,x) atan2a(y,x) 
//#endif
///#ifndef asin(x)
//#define asin(x) asina(x)
//#endif
//#ifndef sinh(x)
//#define sinh(x) sinha(x)
//#endif
//#ifndef tanh(x)
//#define tanh(x) tanha(x)
//#endif
//#ifndef acos(x)
//#define acos(x) acosa(x)
//#endif
//#ifndef ceil(x)
//#define ceil(x) ceila(x)
//#endif
//#ifndef floor(x)
//#define floor(x) floora(x)
//#endif
//#ifndef (y,x)
//#define min(y,x) mina(y,x)
//#endif
//#ifndef max(y,x)
//#define max(y,x) maxa(y,x)
//#endif
//#ifndef fmod(y,x)
//#define fmod(y,x) fmoda(y,x)
//#endif
//#ifndef log10(x)
//#define log10(x) log10a(x)
//#endif
//#ifndef rsqrt(x)
//#define rsqrt(x) rsqrta(x)
//#endif

//double cos(double x) __attribute__ ((alias("cosa")));
//double sin(double x) __attribute__ ((alias("sina")));
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


//#pragma GCC push_options
//#pragma GCC optimize ("-march=core-avx-i -lm -DHAVE_ACML -L/opt/sse -L/opt/OpenBLAS/lib/ -L/usr/local/lib/ -lmingw32 -mf16c -latlas -lfftw3 -llapack -lblas -lOpenCL -lcufft_static -lcublas_static -lclblast -lacml_mv -lamac64o -fopenmp -I/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/include -L/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64 -lmkl_core -lmkl_tbb_thread -lmkl_intel_ilp64 -lstdc++ -lpthread -O1 -funroll-loops -flto -ftree-vectorize -funsafe-math-optimizations -mveclibabi=svml -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mrtm -fipa-cp -ftree-loop-distribution -fprefetch-loop-arrays -freorder-blocks-and-partition -fif-conversion -ftree-loop-if-convert -freorder-blocks -ftree-loop-if-convert -ftree-vectorize -fweb -ftree-loop-if-convert -fsplit-wide-types -ftree-slp-vectorize -ftree-dse -fgcse-las -fsched-dep-count-heuristic -fno-tree-slsr -fsched-spec-load -fconserve-stack -fstrict-aliasing -free -ftree-vrp -fthread-jumps --param=max-reload-search-insns=356 --param=max-cselib-memory-locations=1202 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

static const double Zero[] = {0.0, -0.0,};
#define packed_double(x) {(x), (x)}

#define M_LOG2E 1.4426950408889634074
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#define M_PI_2      1.57079632679489661923132169163975144  
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

// cache last sin(x)
sincache[2] = { 0, 0 };
coscache[2] = { 0, 0 };
expcache[2] = { 0, 0 };
logcache[2] = { 0, 0 };
/*
inline float f32tof16(float x)
{
    //float x;
    unsigned short f16;
    f16 = _cvtss_sh(x, 0);
    return f16;
}
*/

inline double log2(double n)
{
    return log(n) * M_LOG2E;
}

float mmul(const float * a, const float * b, float * r)
{
	__m128 a_line, b_line, r_line;
	int i, j;
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
	m = (int) (ax.f * __sincosf_rng[0]);	
	o = (int) (bx.f * __sincosf_rng[0]);	
	ax.f = ax.f - (((float)m) * __sincosf_rng[1]);
	bx.f = bx.f - (((float)o) * __sincosf_rng[1]);
	
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

float fmoda(float x, float y)
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
	
	n = (int)(x * yinv.f);
	x = x - ((float)n) * y;
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

float log10a(float x)
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
	r.f = r.f + ((float) m) * __log10f_rng;

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
	if(sincache[0]==x)
	{
	return (sincache[1]);
	}
  	if(x<0.3)
	{
	return(x);
	}
  	if(0==fmod(x,90))
	{
	return(0);
	}
	double result = -1.0;
	sin_sse(x, x, &result, &result);
	sincache[0]=x;
	sincache[1]=result;
	return result;
}

double cosaa(double x)
{
	return sina(x + M_PI_2);
}


double ceila(double x)
{
        int i = (int)x;
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
    for ( int i = 0; i < n/4*4; i += 4 ) {
        __m128i vect = _mm_loadu_si128( (__m128i*) (a+i) );
        sum = _mm_add_epi32( vect, sum );
    }
    _mm_storeu_si128( (__m128i*) val, sum);
    int res = val[0] + val[1] + val[2] + val[3];
    for( int i = n/4*4; i < n; i += 1 )
        res += a[i];
    return res;
}

float rcpNewtonRaphsona(float inX, float inRcpX)
{
	return inRcpX * (-inRcpX * inX + 2.0f);
}

inline float acosa(float inX)
{
	const float i[6] = { -0.0187293f,-0.2121144f,0.0742610f ,1.5707288f,1.0f,0.0f  };
	float x1 = abs(inX);
	float x2 = x1 * x1;
	float x3 = x2 * x1;
	float s;

	s = i[1] * x1 + i[3];
	s = i[2] * x2 + s;
	s = i[0] * x3 + s;
	s = sqrt(i[4] - x1) * s;
	return inX >= i[5] ? s : fsl_PI - s;
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

inline double expa(double a)
{
        if(expcache[0]==a)
        {
        return (expcache[1]);
        }
	if(a<0.2)
	{
	return (1+a);
	}
	const int i[2] = { 1512775,1072632447 };
	__builtin_prefetch(&i,1,1);
	union { double d; int x[2]; } u;
	u.x[1] = (int) (i[0] * a + i[1]);
	u.x[0] = 0;
	expcache[0]=a;
	expcache[1]=u.d;
	return u.d;
}

double loga(double a) 
{
	if(logcache[0]==a)
	{
	return logcache[1];
	}
	double y=0;
	union { double d; long long x; } u = { a };
	y = (u.x - 4607182418800017409) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
	logcache[0]=a;
	logcache[1]=y;
	return y;
}


inline double powa(double a, double b) 
{
	union { double d; int x[2]; } u = { a };
	u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
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
return 1.0 / rsqrta(x);
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


double ssin(double x)
{
double b=0;
b=(1-cosa(2*x))/2;
return (b);
}

double scos(double x)
{
double b=0;
b=(1+cosa(2*x))/2;
return (b);
}

double dsin(double x)
{
return sina(2*x);
}

double dcos(double x)
{
return cosa(2*x);
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

double tana(double x)
{
	double input = x;
	return (sin(x)/cos(x));
}

double cota(double x)
{
  double input = x;
  return (cosa(x)/sina(x));
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

//#if defined(LIBM_ALIAS)
/* include aliases to the equivalent libm functions for use with LD_PRELOAD. */ 
//double log(double x) __attribute__ ((alias("loga"))); 
//double exp(double x) __attribute__ ((alias("fm_exp"))); 
//double exp10(double x) __attribute__ ((alias("fm_exp10"))); 
//float exp2f(float x) __attribute__ ((alias("fm_exp2f"))); 
//float expf(float x) __attribute__ ((alias("fm_expf"))); 
//float exp10f(float x) __attribute__ ((alias("fm_exp10f")));

//#pragma GCC pop_options
#endif




