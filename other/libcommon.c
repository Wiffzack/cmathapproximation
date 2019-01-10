#ifndef sseinclude
#define sseinclude
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#ifndef libcom
#define libcom
int convertx(float x)
{
    return _mm_cvtt_ss2si(_mm_load_ss(&x)); // extra 't' means truncate
}

float min(float v0, float v1)
{
        return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(v0), _mm_set_ss(v1)));
}

float max(float v0,float v1)
{
        return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(v0), _mm_set_ss(v1)));
}

float rcp(float op)
{
	return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(op)));
}

float abs(float x)
{
        int casted = *(int*) &x;
        casted &= 0x7FFFFFFF;
        return *(float*)&casted;
}

double ceil(double x)
{
        int i = convertx(x);
        return (x > (i)) ? i+1 : i;
}

static float floor(float x)
{
	__m128 f = _mm_set_ss(x);
	__m128 one = _mm_set_ss(1.0f);
	__m128 t = _mm_cvtepi32_ps(_mm_cvttps_epi32(f));
	__m128 r = _mm_sub_ps(t, _mm_and_ps(_mm_cmplt_ps(f, t), one));
	return _mm_cvtss_f32(r);
}

float sqrt(float op) 
{
	__m128 s = _mm_sqrt_ss(_mm_set_ss(op));
	float r; 
	_mm_store_ss(&r, s);
	return r; 
}


float signnz(float op)
{
        __m128 v = _mm_set_ss(op);
        __m128 s = _mm_or_ps(_mm_and_ps(v, _mm_set_ss(-0.0f)), _mm_set_ss(1.0f));
        return _mm_cvtss_f32(s);
}

float sign(float op)
{
        __m128 v = _mm_set_ss(op);
        __m128 s = _mm_or_ps(_mm_and_ps(v, _mm_set_ss(-0.0f)), _mm_set_ss(1.0f));
        __m128 nz = _mm_cmpneq_ps(v, _mm_setzero_ps());
        __m128 s3 = _mm_and_ps(s, nz);
        return _mm_cvtss_f32(s3);
}
#endif
