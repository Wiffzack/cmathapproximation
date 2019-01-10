# Requirements 
# https://github.com/skywind3000/FastMemcpy skywind3000 and his nice memcpy
# Intel® MPI Library (libmpicxx.a and libmpifort.a)
#
# gcc -c -fPIC memcpy.c -o memcpy.o
# gcc -shared memopt.o -o memopt.so -msse4 -lmpicxx -lmpifort
# export LD_PRELOAD=memopt.so
# pray...


#ifndef fmemcpy
#define fmemcpy
#ifndef sseinclude
#include <smmintrin.h>
#include <immintrin.h>
#endif
//#include <asmlib.h>
#include <FastMemcpy.h>
//#include <FastMemcpy_Avx.h>


extern void *__I_MPI___intel_avx_rep_memcpy(void *str1, const void *str2, size_t n);
extern void *__I_MPI__intel_fast_memcpy(void *str1, const void *str2, size_t n);
//extern void *A_memcpy(void *str1, const void *str2, size_t n);
//extern void *memcpy(void *str1, const void *str2, size_t n);

//extern void *A_memmove(void *str1, const void *str2, size_t n);
extern void *__I_MPI__intel_fast_memmove(void *str1, const void *str2, size_t n);


void* memcpy(void *dst, const void *src, size_t size)
{
	if (size <= 128)
	{
		return memcpy_tiny(dst, src, size);
	}
	if(size > 128 && size <= 512)
	{
		return __I_MPI___intel_avx_rep_memcpy(dst, src, size);
	}
	if(size > 512 && size <= 1024)
	{
		return __I_MPI__intel_fast_memcpy(dst, src, size);
	}
	if (size > 1024 && size <= 8192)
	{
		return memcpy_fast(dst, src, size);
	}
	if( size > 8192 && size <= 1048576 )
	{
		return __I_MPI__intel_fast_memcpy(dst, src, size);
	}
	if( size > 1048576 )
	{
		return __I_MPI___intel_avx_rep_memcpy(dst, src, size);
	}
}

#endif
