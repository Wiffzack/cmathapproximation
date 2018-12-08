#include <smmintrin.h>
#include <immintrin.h>

#pragma GCC push_options
#pragma GCC optimize (" -fno-builtin-strlen  -fno-builtin-strstr  -fno-builtin-strrchr  -fno-builtin-memset  -fno-builtin-memcpy ")

char lower_digits[] = "0123456789";

#define INTBITS (sizeof(int)*8)
static inline void setbit(void *v, int p)          // p in 0..255
{
    ((int *)v)[p / INTBITS] |= 1 << (p & (INTBITS - 1));
}

__m128i tovector(char * c) {
  char buffer[16];
  strncpy(buffer,c,16);
  return _mm_loadu_si128((const __m128i *)buffer);
}


#define array_elements(array) (sizeof(array) / sizeof *(array))

#pragma GCC push_options
#pragma GCC optimize ("-D_FORTIFY_SOURCE=1 /usr/lib/strcmp32.o /usr/lib/strcmp.o")
extern int strcmp32(const char *, const char *);
extern int strcmp(const char *, const char *);
//static int (*strcmp)(const char *, const char *) = strcmp;
#pragma GCC pop_options

static void __memset(void *ptr, int c, size_t len)
{
    unsigned int i;
    register char *p = (char *) ptr;
    #pragma omp parallel for
    for (i = 0; i < len; i++) {
        p[i] = c;
    }
}

void memset(void *ptr, int v, size_t len)
{
    register size_t i;
    size_t loff = ((intptr_t) ptr) % 16;
    size_t l16 = (len - loff) / 16;
    size_t lrem = len - l16 * 16 - loff;
    register char *p = (char *) ptr;
    char  c = (char) v;
    __m128i c16 = _mm_set_epi8(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
    __memset(p, c, loff);
    p += loff;
    for (i = 0; i < l16; ++i) {
        _mm_store_si128((__m128i*)p, c16);
        p += 16;
    }
    __memset(p, c, lrem);
}


size_t strlen(const char* top)
{
        const __m128i im = _mm_set1_epi32(0xff01);
        const char *p = top;
        while (!_mm_cmpistrz(im, _mm_loadu_si128((const __m128i*)p), 0x14)) {
                p += 16;
        }
        p += _mm_cmpistri(im, _mm_loadu_si128((const __m128i*)p), 0x14);
        return p - top;
}

char* strrchr(const char* s, int c) {

    __m128i* mem = (__m128i*)((char*)(s));
    const __m128i set = _mm_setr_epi8(c, 0, 0, 0, 0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0);

    const uint8_t mode = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT;

    char* result = NULL;
    #pragma omp parallel for
    for (/**/; /**/; mem++) {

        const __m128i chunk = _mm_loadu_si128(mem);

        if (_mm_cmpistrc(set, chunk, mode)) {
            // there is character c in a chunk
            const auto idx = _mm_cmpistri(set, chunk, mode);

            result = (char*)(mem) + idx;
        } else if (_mm_cmpistrz(set, chunk, mode)) {
            // there is zero byte in a chunk
            break;
        }
    }

    return result;
}

char* strchr(const char* s, int c) {

    __m128i* mem = (__m128i*)((char*)(s));
    const __m128i set = _mm_setr_epi8(c, 0, 0, 0, 0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0);

    const uint8_t mode =
        _SIDD_UBYTE_OPS |
        _SIDD_CMP_EQUAL_ANY |
        _SIDD_LEAST_SIGNIFICANT;

    #pragma omp parallel for
    for (/**/; /**/; mem++) {

        const __m128i chunk = _mm_loadu_si128(mem);

        if (_mm_cmpistrc(set, chunk, mode)) {
            // there is character c in a chunk
            const auto idx = _mm_cmpistri(set, chunk, mode);

            return (char*)(mem) + idx;
        } else if (_mm_cmpistrz(set, chunk, mode)) {
            // there is zero byte in a chunk
            break;
        }
    }

    return NULL;
}

int memcmp(void* s1, void* s2, size_t n) {

    if (n == 0 || (s1 == s2)) {
        return 0;
    }

    __m128i* ptr1 = (__m128i*)(s1);
    __m128i* ptr2 = (__m128i*)(s2);

    const uint8_t mode =
        _SIDD_UBYTE_OPS |
        _SIDD_CMP_EQUAL_EACH |
        _SIDD_NEGATIVE_POLARITY |
        _SIDD_LEAST_SIGNIFICANT;

    #pragma omp parallel for
    for (/**/; n != 0; ptr1++, ptr2++) {

        const __m128i a = _mm_loadu_si128(ptr1);
        const __m128i b = _mm_loadu_si128(ptr2);

        if (_mm_cmpestrc(a, n, b, n, mode)) {
            
            const auto idx = _mm_cmpestri(a, n, b, n, mode);

            const uint8_t b1 = ((char*)(ptr1))[idx];
            const uint8_t b2 = ((char*)(ptr2))[idx];

            if (b1 < b2) {
                return -1;
            } else if (b1 > b2) {
                return +1;
            } else {
                return 0;
            }
        } 

        if (n > 16) {
            n -= 16;
        } else {
            n = 0;
        }
    }

    return 0;
}

//test 

int strstr(char * hay, int size, char *needle, int needlesize) {
  __m128i n = tovector(needle);
  #pragma omp parallel for
  for(unsigned int i = 0; i + 15 < size; i+=16) {
    __m128i v = _mm_loadu_si128((const __m128i *)(hay + i));
    __m128i x = _mm_cmpistrm(n,v,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_BIT_MASK);
    int r = _mm_cvtsi128_si32(x);
    if(r != 0) {
      int offset = __builtin_ctz(r);
      v = _mm_loadu_si128((const __m128i *)(hay + i + offset));
      x = _mm_cmpistrm(n,v,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_BIT_MASK);
      r = _mm_cvtsi128_si32(x);
      if((r & 1 ) != 0) return i + offset;
    }
  }
  return size;
}




void *sse_store(void *dst, const void *src, size_t sz)
{
    size_t sz_sse = sz/16;
    __m128 *vdst = dst;
    //const volatile __m128 *vsrc = src;

    for (size_t i=0; i<sz_sse; i+=8) {
        vdst[i+0] = _mm_setzero_ps();
        vdst[i+1] = _mm_setzero_ps();
        vdst[i+2] = _mm_setzero_ps();
        vdst[i+3] = _mm_setzero_ps();

        vdst[i+4] = _mm_setzero_ps();
        vdst[i+5] = _mm_setzero_ps();
        vdst[i+6] = _mm_setzero_ps();
        vdst[i+7] = _mm_setzero_ps();
    }
    return NULL;
}

void *sse_stream_store(void *dst, const void *src, size_t sz)
{
    size_t sz_sse = sz/16;
    __m128 *vdst = dst;
    //const volatile __m128 *vsrc = src;

    for (size_t i=0; i<sz_sse; i+=4) {
        _mm_stream_ps((float*)&vdst[i+0], _mm_setzero_ps());
        _mm_stream_ps((float*)&vdst[i+1], _mm_setzero_ps());
        _mm_stream_ps((float*)&vdst[i+2], _mm_setzero_ps());
        _mm_stream_ps((float*)&vdst[i+3], _mm_setzero_ps());
    }
    return NULL;
}

void *rep_movsb(void *dst, const void *src, size_t sz)
{
    asm volatile ("rep movsb"
                  :"=D" (dst), "=S" (src), "=c" (sz)
                  : "0" (dst), "1" (src), "2" (sz)
                  : "memory");
    return NULL;
}

void *rep_stosb(void *dst, const void *src, size_t sz)
{
    asm volatile ("rep stosb"
                  :"=D" (dst), "=S" (src), "=c" (sz)
                  : "0" (dst), "1" (src), "2" (sz)
                  : "memory");
    return NULL;
}



/*
char   *
ssebndm(char *target, int tgtlen, char *pattern, int patlen)
{
    uint8_t *tgt = (uint8_t *) target, *pat = (uint8_t *) pattern;
    uint8_t *tgtend = tgt + tgtlen - patlen;
    int     i, skip;

#if __SIZEOF_INT__ == 4
    if (patlen <= 32) {
        int32_t mask, maskv[256] = { };

        for (i = 0; i < patlen; ++i)
            setbit(&maskv[pat[i]], sizeof(mask) * 8 - 1 - i);

        for (; tgt <= tgtend; tgt += skip)
            for (i = skip = patlen, mask = maskv[tgt[--i]];
                 mask; mask = (mask << 1) & maskv[tgt[--i]])
                if (mask < 0 && !(skip = i))
                    return (char *)tgt;
    } else
#endif
    if (patlen <= 64) {
        int64_t mask, maskv[256] = { };

        for (i = 0; i < patlen; ++i)
            setbit(&maskv[pat[i]], sizeof(mask) * 8 - 1 - i);

        for (; tgt <= tgtend; tgt += skip)
            for (i = skip = patlen, mask = maskv[tgt[--i]];
                 mask; mask = (mask << 1) & maskv[tgt[--i]])
                if (mask < 0 && !(skip = i))
                    return (char *)tgt;

    } else if (patlen <= 128) {
        __m128i mask, maskv[256], zero = { };
        int8_t  used[256] = { };

        for (i = 0; i < patlen; ++i) {
            if (!used[pat[i]])
                used[pat[i]] = 1, maskv[pat[i]] = zero;
            setbit(&maskv[pat[i]], sizeof(mask) * 8 - 1 - i);
        }

        for (; tgt <= tgtend; tgt += skip) {
            i = skip = patlen;
            if (!used[tgt[--i]])
                continue;
            mask = maskv[tgt[i]];   // gteed not zero.
            do {
                if (0 > (int16_t) _mm_movemask_epi8(mask) && !(skip = i))
                    return (char *)tgt;
                if (!used[tgt[--i]])
                    break;
                mask = _mm_and_si128(xm_shl_001(mask), maskv[tgt[i]]);
            } while (0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8(mask, zero)));
        }
    }
    //TODO: AVX 256-bit ops.
    return NULL;
}
*/




/*
int ltoa_sse4 (const long long s1, char * buf)
{
	long long temp ;
	int j = 1, len = 0;
	const char *digits = &lower_digits[0];
	if( s1 < 0) {
		temp = -s1;
		len ++;
 		beg[0] = '-';
		if( temp < 10) 
			beg[1] = digits[ (int) temp];
		else 
			len += sse4i_q2a_u63( temp, &buf[ 1]); // parallel conversion in 4-digit granular operation
		}
		else 
		{
		if( s1 < 10)
			beg[ 0 ] = digits[(int)s1];
		else 
			len += sse4i_q2a_u63( s1, &buf[ 1] );
		}
		buf[len] = 0;
		return len;
}



void KMPSearch(char* pat, char* txt) 
{ 
    int M = strlen(pat); 
    int N = strlen(txt); 
  
    // create lps[] that will hold the longest prefix suffix 
    // values for pattern 
    int lps[M]; 
  
    // Preprocess the pattern (calculate lps[] array) 
    computeLPSArray(pat, M, lps); 
  
    int i = 0; // index for txt[] 
    int j = 0; // index for pat[] 
    while (i < N) { 
        if (pat[j] == txt[i]) { 
            j++; 
            i++; 
        } 
  
        if (j == M) { 
            printf("Found pattern at index %d ", i - j); 
            j = lps[j - 1]; 
        } 
  
        // mismatch after j matches 
        else if (i < N && pat[j] != txt[i]) { 
            // Do not match lps[0..lps[j-1]] characters, 
            // they will match anyway 
            if (j != 0) 
                j = lps[j - 1]; 
            else
                i = i + 1; 
        } 
    } 
} 
  
// Fills lps[] for given patttern pat[0..M-1] 
void computeLPSArray(char* pat, int M, int* lps) 
{ 
    // length of the previous longest prefix suffix 
    int len = 0; 
  
    lps[0] = 0; // lps[0] is always 0 
  
    // the loop calculates lps[i] for i = 1 to M-1 
    int i = 1; 
    while (i < M) { 
        if (pat[i] == pat[len]) { 
            len++; 
            lps[i] = len; 
            i++; 
        } 
        else // (pat[i] != pat[len]) 
        { 
            // This is tricky. Consider the example. 
            // AAACAAAA and i = 7. The idea is similar 
            // to search step. 
            if (len != 0) { 
                len = lps[len - 1]; 
  
                // Also, note that we do not increment 
                // i here 
            } 
            else // if (len == 0) 
            { 
                lps[i] = 0; 
                i++; 
            } 
        } 
    } 
} */
  
#pragma GCC pop_options
