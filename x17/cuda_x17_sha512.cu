/**
* sha512 djm34
* (cleaned by tpruvot)
*/

/*
* sha-512 kernel implementation.
*
* ==========================(LICENSE BEGIN)============================
*
* Copyright (c) 2014 djm34
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* ===========================(LICENSE END)=============================
*
* @author phm <phm@inbox.com>
*/
#include <stdio.h>
#include <memory.h>
#define USE_SHARED 1
#define SPH_C64(x) ((uint64_t)(x ## ULL))

#include "cuda_helper.h"

__constant__  uint32_t pTarget[8];
__constant__  uint32_t  c_data[48];
static uint32_t *d_found[MAX_GPUS];


#define SWAP64(u64) cuda_swab64(u64)

#define SPH_ROTL32(x, n)  ROTL32(x, n)
#define SPH_ROTR32(x, n)  ROTR32(x, n)

static __constant__ uint64_t H_512[8] = {
	SPH_C64(0x6A09E667F3BCC908), SPH_C64(0xBB67AE8584CAA73B),
	SPH_C64(0x3C6EF372FE94F82B), SPH_C64(0xA54FF53A5F1D36F1),
	SPH_C64(0x510E527FADE682D1), SPH_C64(0x9B05688C2B3E6C1F),
	SPH_C64(0x1F83D9ABFB41BD6B), SPH_C64(0x5BE0CD19137E2179)
};

static __constant__ uint64_t K_512[80] = {
	SPH_C64(0x428A2F98D728AE22), SPH_C64(0x7137449123EF65CD),
	SPH_C64(0xB5C0FBCFEC4D3B2F), SPH_C64(0xE9B5DBA58189DBBC),
	SPH_C64(0x3956C25BF348B538), SPH_C64(0x59F111F1B605D019),
	SPH_C64(0x923F82A4AF194F9B), SPH_C64(0xAB1C5ED5DA6D8118),
	SPH_C64(0xD807AA98A3030242), SPH_C64(0x12835B0145706FBE),
	SPH_C64(0x243185BE4EE4B28C), SPH_C64(0x550C7DC3D5FFB4E2),
	SPH_C64(0x72BE5D74F27B896F), SPH_C64(0x80DEB1FE3B1696B1),
	SPH_C64(0x9BDC06A725C71235), SPH_C64(0xC19BF174CF692694),
	SPH_C64(0xE49B69C19EF14AD2), SPH_C64(0xEFBE4786384F25E3),
	SPH_C64(0x0FC19DC68B8CD5B5), SPH_C64(0x240CA1CC77AC9C65),
	SPH_C64(0x2DE92C6F592B0275), SPH_C64(0x4A7484AA6EA6E483),
	SPH_C64(0x5CB0A9DCBD41FBD4), SPH_C64(0x76F988DA831153B5),
	SPH_C64(0x983E5152EE66DFAB), SPH_C64(0xA831C66D2DB43210),
	SPH_C64(0xB00327C898FB213F), SPH_C64(0xBF597FC7BEEF0EE4),
	SPH_C64(0xC6E00BF33DA88FC2), SPH_C64(0xD5A79147930AA725),
	SPH_C64(0x06CA6351E003826F), SPH_C64(0x142929670A0E6E70),
	SPH_C64(0x27B70A8546D22FFC), SPH_C64(0x2E1B21385C26C926),
	SPH_C64(0x4D2C6DFC5AC42AED), SPH_C64(0x53380D139D95B3DF),
	SPH_C64(0x650A73548BAF63DE), SPH_C64(0x766A0ABB3C77B2A8),
	SPH_C64(0x81C2C92E47EDAEE6), SPH_C64(0x92722C851482353B),
	SPH_C64(0xA2BFE8A14CF10364), SPH_C64(0xA81A664BBC423001),
	SPH_C64(0xC24B8B70D0F89791), SPH_C64(0xC76C51A30654BE30),
	SPH_C64(0xD192E819D6EF5218), SPH_C64(0xD69906245565A910),
	SPH_C64(0xF40E35855771202A), SPH_C64(0x106AA07032BBD1B8),
	SPH_C64(0x19A4C116B8D2D0C8), SPH_C64(0x1E376C085141AB53),
	SPH_C64(0x2748774CDF8EEB99), SPH_C64(0x34B0BCB5E19B48A8),
	SPH_C64(0x391C0CB3C5C95A63), SPH_C64(0x4ED8AA4AE3418ACB),
	SPH_C64(0x5B9CCA4F7763E373), SPH_C64(0x682E6FF3D6B2B8A3),
	SPH_C64(0x748F82EE5DEFB2FC), SPH_C64(0x78A5636F43172F60),
	SPH_C64(0x84C87814A1F0AB72), SPH_C64(0x8CC702081A6439EC),
	SPH_C64(0x90BEFFFA23631E28), SPH_C64(0xA4506CEBDE82BDE9),
	SPH_C64(0xBEF9A3F7B2C67915), SPH_C64(0xC67178F2E372532B),
	SPH_C64(0xCA273ECEEA26619C), SPH_C64(0xD186B8C721C0C207),
	SPH_C64(0xEADA7DD6CDE0EB1E), SPH_C64(0xF57D4F7FEE6ED178),
	SPH_C64(0x06F067AA72176FBA), SPH_C64(0x0A637DC5A2C898A6),
	SPH_C64(0x113F9804BEF90DAE), SPH_C64(0x1B710B35131C471B),
	SPH_C64(0x28DB77F523047D84), SPH_C64(0x32CAAB7B40C72493),
	SPH_C64(0x3C9EBE0A15C9BEBC), SPH_C64(0x431D67C49C100D4C),
	SPH_C64(0x4CC5D4BECB3E42B6), SPH_C64(0x597F299CFC657E2A),
	SPH_C64(0x5FCB6FAB3AD6FAEC), SPH_C64(0x6C44198C4A475817)
};

//#define BSG5_0(x)      (ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define BSG5_0(x)        xor3(ROTR64(x, 28),ROTR64(x, 34),ROTR64(x, 39))

//#define BSG5_1(x)      (ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define BSG5_1(x)      xor3(ROTR64(x, 14),ROTR64(x, 18),ROTR64(x, 41))

//#define SSG5_0(x)      (ROTR64(x, 1) ^ ROTR64(x, 8) ^ SPH_T64((x) >> 7))
#define SSG5_0(x)      xor3(ROTR64(x, 1),ROTR64(x, 8),shr_t64(x,7))

//#define SSG5_1(x)      (ROTR64(x, 19) ^ ROTR64(x, 61) ^ SPH_T64((x) >> 6))
#define SSG5_1(x)      xor3(ROTR64(x, 19),ROTR64(x, 61),shr_t64(x,6))

//#define CH(X, Y, Z)    ((((Y) ^ (Z)) & (X)) ^ (Z))
#define CH(x, y, z)    xandx(x,y,z)
//#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
#define MAJ(x, y, z)   andor(x,y,z)

#define SHA3_STEP(ord,r,i) { \
		uint64_t T1, T2; \
		int a = 8-ord; \
		T1 = r[(7+a)&7] + BSG5_1(r[(4+a)&7]) + CH(r[(4+a)&7], r[(5+a)&7], r[(6+a)&7]) + K_512[i] + W[i]; \
		T2 = BSG5_0(r[(0+a)&7]) + MAJ(r[(0+a)&7], r[(1+a)&7], r[(2+a)&7]); \
		r[(3+a)&7] = r[(3+a)&7] + T1; \
		r[(7+a)&7] = T1 + T2; \
	}

__device__ __forceinline__
uint64_t Tone(const uint64_t* sharedMemory, uint64_t r[8], uint64_t W[80], uint32_t a, uint32_t i)
{
	uint64_t e = r[(4 + a) & 7];
	//uint64_t BSG51 = ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41);
	uint64_t BSG51 = xor3(ROTR64(e, 14), ROTR64(e, 18), ROTR64(e, 41));

	//uint64_t CHl     = (((f) ^ (g)) & (e)) ^ (g);
	uint64_t CHl = xandx(e, r[(5 + a) & 7], r[(6 + a) & 7]);
	uint64_t result = r[(7 + a) & 7] + BSG51 + CHl + sharedMemory[i] + W[i];
	return result;
}

#define SHA3_STEP2(truc,ord,r,i) { \
		uint64_t T1, T2; \
		int a = 8-ord; \
		T1 = Tone(truc,r,W,a,i); \
		T2 = BSG5_0(r[(0+a)&7]) + MAJ(r[(0+a)&7], r[(1+a)&7], r[(2+a)&7]); \
		r[(3+a)&7] = r[(3+a)&7] + T1; \
		r[(7+a)&7] = T1 + T2; \
	}

#define TPB 128
__global__ __launch_bounds__(TPB, 6)
void x17_sha512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	//	if (thread < threads)
	{
		uint64_t *inpHash = &g_hash[8 * thread];
		uint64_t hash[8];

#pragma unroll
		for (int i = 0; i<8; i++)
		{
			hash[i] = inpHash[i];
		}
		uint64_t W[80] = { 0 };
		uint64_t r[8];

#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			W[i] = SWAP64(hash[i]);
			r[i] = H_512[i];
		}

		W[8] = 0x8000000000000000;
		W[15] = 0x0000000000000200;

#pragma unroll 64
		for (int i = 16; i < 80; i++)
			W[i] = SSG5_1(W[i - 2]) + W[i - 7] + SSG5_0(W[i - 15]) + W[i - 16];

#pragma unroll 10
		for (int i = 0; i < 80; i += 8)
		{
#pragma unroll 8
			for (int ord = 0; ord<8; ord++)
			{
				SHA3_STEP2(K_512, ord, r, i + ord);
			}
		}

#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			r[i] = r[i] + H_512[i];
		}

#pragma unroll 8
		for (int i = 0; i<8; i++)
		{
			hash[i] = SWAP64(r[i]);
		}

#pragma unroll 16
		for (int u = 0; u < 8; u++)
		{
			inpHash[u] = hash[u];
		}
	}
}
__host__
void x17_sha512_cpu_init(int thr_id, uint32_t threads)
{
//	cudaMemcpyToSymbol(K_512,K512,80*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(H_512,H512,sizeof(H512),0, cudaMemcpyHostToDevice);
}

__host__
void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	x17_sha512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash );
}

static const __constant__ uint64_t K512[80] =
{
	0x428A2F98D728AE22UL, 0x7137449123EF65CDUL,
	0xB5C0FBCFEC4D3B2FUL, 0xE9B5DBA58189DBBCUL,
	0x3956C25BF348B538UL, 0x59F111F1B605D019UL,
	0x923F82A4AF194F9BUL, 0xAB1C5ED5DA6D8118UL,
	0xD807AA98A3030242UL, 0x12835B0145706FBEUL,
	0x243185BE4EE4B28CUL, 0x550C7DC3D5FFB4E2UL,
	0x72BE5D74F27B896FUL, 0x80DEB1FE3B1696B1UL,
	0x9BDC06A725C71235UL, 0xC19BF174CF692694UL,
	0xE49B69C19EF14AD2UL, 0xEFBE4786384F25E3UL,
	0x0FC19DC68B8CD5B5UL, 0x240CA1CC77AC9C65UL,
	0x2DE92C6F592B0275UL, 0x4A7484AA6EA6E483UL,
	0x5CB0A9DCBD41FBD4UL, 0x76F988DA831153B5UL,
	0x983E5152EE66DFABUL, 0xA831C66D2DB43210UL,
	0xB00327C898FB213FUL, 0xBF597FC7BEEF0EE4UL,
	0xC6E00BF33DA88FC2UL, 0xD5A79147930AA725UL,
	0x06CA6351E003826FUL, 0x142929670A0E6E70UL,
	0x27B70A8546D22FFCUL, 0x2E1B21385C26C926UL,
	0x4D2C6DFC5AC42AEDUL, 0x53380D139D95B3DFUL,
	0x650A73548BAF63DEUL, 0x766A0ABB3C77B2A8UL,
	0x81C2C92E47EDAEE6UL, 0x92722C851482353BUL,
	0xA2BFE8A14CF10364UL, 0xA81A664BBC423001UL,
	0xC24B8B70D0F89791UL, 0xC76C51A30654BE30UL,
	0xD192E819D6EF5218UL, 0xD69906245565A910UL,
	0xF40E35855771202AUL, 0x106AA07032BBD1B8UL,
	0x19A4C116B8D2D0C8UL, 0x1E376C085141AB53UL,
	0x2748774CDF8EEB99UL, 0x34B0BCB5E19B48A8UL,
	0x391C0CB3C5C95A63UL, 0x4ED8AA4AE3418ACBUL,
	0x5B9CCA4F7763E373UL, 0x682E6FF3D6B2B8A3UL,
	0x748F82EE5DEFB2FCUL, 0x78A5636F43172F60UL,
	0x84C87814A1F0AB72UL, 0x8CC702081A6439ECUL,
	0x90BEFFFA23631E28UL, 0xA4506CEBDE82BDE9UL,
	0xBEF9A3F7B2C67915UL, 0xC67178F2E372532BUL,
	0xCA273ECEEA26619CUL, 0xD186B8C721C0C207UL,
	0xEADA7DD6CDE0EB1EUL, 0xF57D4F7FEE6ED178UL,
	0x06F067AA72176FBAUL, 0x0A637DC5A2C898A6UL,
	0x113F9804BEF90DAEUL, 0x1B710B35131C471BUL,
	0x28DB77F523047D84UL, 0x32CAAB7B40C72493UL,
	0x3C9EBE0A15C9BEBCUL, 0x431D67C49C100D4CUL,
	0x4CC5D4BECB3E42B6UL, 0x597F299CFC657E2AUL,
	0x5FCB6FAB3AD6FAECUL, 0x6C44198C4A475817UL
};

static const __constant__ uint64_t SHA512_INIT[8] =
{
	0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
	0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
	0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
	0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
};

//#define ROTR64(x, y)	rotate((x), 64UL - (y))
#define nvidia_bitalign(src0,src1,src2) (((src0) << (src2)) | ((src1) >> (32-(src2))))


//uint64_t FAST_ROTR64_LO(const uint2 x, const uint32_t y) { return(devectorize(nvidia_bitalign(x.y, x.x, y))); }
//uint64_t FAST_ROTR64_HI(const uint2 x, const uint32_t y) { return(devectorize(nvidia_bitalign(x.x, x.y, (y - 32)))); }

/*
#define BSG5_0(x) (FAST_ROTR64_LO(x, 28) ^ FAST_ROTR64_HI(x, 34) ^ FAST_ROTR64_HI(x, 39))
#define BSG5_1(x) (FAST_ROTR64_LO(x, 14) ^ FAST_ROTR64_LO(x, 18) ^ ROTR64(x, 41))
#define SSG5_0(x) (FAST_ROTR64_LO(x, 1) ^ FAST_ROTR64_LO(x, 8) ^ ((x) >> 7))
#define SSG5_1(x) (FAST_ROTR64_LO(x, 19) ^ FAST_ROTR64_HI(x, 61) ^ ((x) >> 6))
*/

//#define BSG5_0(x) (FAST_ROTR64_LO(as_uint2(x), 28) ^ FAST_ROTR64_HI(as_uint2(x), 34) ^ FAST_ROTR64_HI(as_uint2(x), 39))
//#define BSG5_1(x) (FAST_ROTR64_LO(as_uint2(x), 14) ^ FAST_ROTR64_LO(as_uint2(x), 18) ^ FAST_ROTR64_HI(as_uint2(x), 41))
//#define SSG5_0(x) (FAST_ROTR64_LO(as_uint2(x), 1) ^ FAST_ROTR64_LO(as_uint2(x), 8) ^ ((x) >> 7))
//#define SSG5_1(x) (FAST_ROTR64_LO(as_uint2(x), 19) ^ FAST_ROTR64_HI(as_uint2(x), 61) ^ ((x) >> 6))

//#define CH(X, Y, Z) bitselect(Z, Y, X)
//#define MAJ(X, Y, Z) CH((X ^ Z), Y, Z)

__device__ __forceinline__ void SHA2_512_STEP2(const uint64_t *W, uint64_t ord, uint64_t *r, int i)
{
	uint64_t T1;
	int x = 8 - ord;

	uint64_t a = r[x & 7], b = r[(x + 1) & 7], c = r[(x + 2) & 7], d = r[(x + 3) & 7];
	uint64_t e = r[(x + 4) & 7], f = r[(x + 5) & 7], g = r[(x + 6) & 7], h = r[(x + 7) & 7];

	T1 = h + BSG5_1(e) + CH(e, f, g) + W[i] + K512[i];
	r[(3 + x) & 7] = d + T1;
	r[(7 + x) & 7] = T1 + BSG5_0(a) + MAJ(a, b, c);
}

__device__ __forceinline__ void SHA512Block(uint64_t *data, uint64_t *buf)
{
	uint64_t W[80], r[8];

	for (int i = 0; i < 8; ++i) r[i] = buf[i];

	for (int i = 0; i < 16; ++i) W[i] = data[i];

#pragma unroll 4
	for (int i = 16; i < 80; ++i) W[i] = SSG5_1(W[i - 2]) + W[i - 7] + SSG5_0(W[i - 15]) + W[i - 16];

#pragma unroll 1
	for (int i = 0; i < 80; i += 8)
	{
#pragma unroll
		for (int j = 0; j < 8; ++j)
		{
			SHA2_512_STEP2(W, j, r, i + j);
		}
	}

	for (int i = 0; i < 8; ++i) buf[i] += r[i];
}


#define RIPEMD160_IN(x) W[x]

// Round functions for RIPEMD-128 and RIPEMD-160.

#define F1(x, y, z)  	((x) ^ (y) ^ (z))
#define F2(x, y, z)   	((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   	(((x) | ~(y)) ^ (z))
#define F4(x, y, z)   	((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   	((x) ^ ((y) | ~(z)))

#define K11    0x00000000
#define K12    0x5A827999
#define K13    0x6ED9EBA1
#define K14    0x8F1BBCDC
#define K15    0xA953FD4E

#define K21    0x50A28BE6
#define K22    0x5C4DD124
#define K23    0x6D703EF3
#define K24    0x7A6D76E9
#define K25    0x00000000

const __constant__ uint32_t RMD160_IV[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };

#define RR(a, b, c, d, e, f, s, r, k)   do { \
		const uint32_t rrtmp = a + f(b, c, d) + r + k; \
		a = nvidia_bitalign(rrtmp, rrtmp, 32U - (uint32_t)s) + e; \
		c = nvidia_bitalign(c, c, 32U - 10U); \
	} while (0)

#define ROUND1(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 1, b ## 1, c ## 1, d ## 1, e ## 1, f, s, r, K1 ## k)

#define ROUND2(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 2, b ## 2, c ## 2, d ## 2, e ## 2, f, s, r, K2 ## k)

/*
* This macro defines the body for a RIPEMD-160 compression function
* implementation. The "in" parameter should evaluate, when applied to a
* numerical input parameter from 0 to 15, to an expression which yields
* the corresponding input block. The "h" parameter should evaluate to
* an array or pointer expression designating the array of 5 words which
* contains the input and output of the compression function.
*/

//#define RIPEMD160_ROUND_BODY(in, h)   do { \
		uint A1, B1, C1, D1, E1; \
		uint A2, B2, C2, D2, E2; \
		uint tmp; \
 \
		A1 = A2 = (h)[0]; \
		B1 = B2 = (h)[1]; \
		C1 = C2 = (h)[2]; \
		D1 = D2 = (h)[3]; \
		E1 = E2 = (h)[4]; \
 \
		ROUND1(A, B, C, D, E, F1, 11, (in)[ 0],  1); \
		ROUND1(E, A, B, C, D, F1, 14, (in)[ 1],  1); \
		ROUND1(D, E, A, B, C, F1, 15, (in)[ 2],  1); \
		ROUND1(C, D, E, A, B, F1, 12, (in)[ 3],  1); \
		ROUND1(B, C, D, E, A, F1,  5, (in)[ 4],  1); \
		ROUND1(A, B, C, D, E, F1,  8, (in)[ 5],  1); \
		ROUND1(E, A, B, C, D, F1,  7, (in)[ 6],  1); \
		ROUND1(D, E, A, B, C, F1,  9, (in)[ 7],  1); \
		ROUND1(C, D, E, A, B, F1, 11, (in)[ 8],  1); \
		ROUND1(B, C, D, E, A, F1, 13, (in)[ 9],  1); \
		ROUND1(A, B, C, D, E, F1, 14, (in)[10],  1); \
		ROUND1(E, A, B, C, D, F1, 15, (in)[11],  1); \
		ROUND1(D, E, A, B, C, F1,  6, (in)[12],  1); \
		ROUND1(C, D, E, A, B, F1,  7, (in)[13],  1); \
		ROUND1(B, C, D, E, A, F1,  9, (in)[14],  1); \
		ROUND1(A, B, C, D, E, F1,  8, (in)[15],  1); \
 \
		ROUND1(E, A, B, C, D, F2,  7, (in)[ 7],  2); \
		ROUND1(D, E, A, B, C, F2,  6, (in)[ 4],  2); \
		ROUND1(C, D, E, A, B, F2,  8, (in)[13],  2); \
		ROUND1(B, C, D, E, A, F2, 13, (in)[ 1],  2); \
		ROUND1(A, B, C, D, E, F2, 11, (in)[10],  2); \
		ROUND1(E, A, B, C, D, F2,  9, (in)[ 6],  2); \
		ROUND1(D, E, A, B, C, F2,  7, (in)[15],  2); \
		ROUND1(C, D, E, A, B, F2, 15, (in)[ 3],  2); \
		ROUND1(B, C, D, E, A, F2,  7, (in)[12],  2); \
		ROUND1(A, B, C, D, E, F2, 12, (in)[ 0],  2); \
		ROUND1(E, A, B, C, D, F2, 15, (in)[ 9],  2); \
		ROUND1(D, E, A, B, C, F2,  9, (in)[ 5],  2); \
		ROUND1(C, D, E, A, B, F2, 11, (in)[ 2],  2); \
		ROUND1(B, C, D, E, A, F2,  7, (in)[14],  2); \
		ROUND1(A, B, C, D, E, F2, 13, (in)[11],  2); \
		ROUND1(E, A, B, C, D, F2, 12, (in)[ 8],  2); \
 \
		ROUND1(D, E, A, B, C, F3, 11, (in)[ 3],  3); \
		ROUND1(C, D, E, A, B, F3, 13, (in)[10],  3); \
		ROUND1(B, C, D, E, A, F3,  6, (in)[14],  3); \
		ROUND1(A, B, C, D, E, F3,  7, (in)[ 4],  3); \
		ROUND1(E, A, B, C, D, F3, 14, (in)[ 9],  3); \
		ROUND1(D, E, A, B, C, F3,  9, (in)[15],  3); \
		ROUND1(C, D, E, A, B, F3, 13, (in)[ 8],  3); \
		ROUND1(B, C, D, E, A, F3, 15, (in)[ 1],  3); \
		ROUND1(A, B, C, D, E, F3, 14, (in)[ 2],  3); \
		ROUND1(E, A, B, C, D, F3,  8, (in)[ 7],  3); \
		ROUND1(D, E, A, B, C, F3, 13, (in)[ 0],  3); \
		ROUND1(C, D, E, A, B, F3,  6, (in)[ 6],  3); \
		ROUND1(B, C, D, E, A, F3,  5, (in)[13],  3); \
		ROUND1(A, B, C, D, E, F3, 12, (in)[11],  3); \
		ROUND1(E, A, B, C, D, F3,  7, (in)[ 5],  3); \
		ROUND1(D, E, A, B, C, F3,  5, (in)[12],  3); \
 \
		ROUND1(C, D, E, A, B, F4, 11, (in)[ 1],  4); \
		ROUND1(B, C, D, E, A, F4, 12, (in)[ 9],  4); \
		ROUND1(A, B, C, D, E, F4, 14, (in)[11],  4); \
		ROUND1(E, A, B, C, D, F4, 15, (in)[10],  4); \
		ROUND1(D, E, A, B, C, F4, 14, (in)[ 0],  4); \
		ROUND1(C, D, E, A, B, F4, 15, (in)[ 8],  4); \
		ROUND1(B, C, D, E, A, F4,  9, (in)[12],  4); \
		ROUND1(A, B, C, D, E, F4,  8, (in)[ 4],  4); \
		ROUND1(E, A, B, C, D, F4,  9, (in)[13],  4); \
		ROUND1(D, E, A, B, C, F4, 14, (in)[ 3],  4); \
		ROUND1(C, D, E, A, B, F4,  5, (in)[ 7],  4); \
		ROUND1(B, C, D, E, A, F4,  6, (in)[15],  4); \
		ROUND1(A, B, C, D, E, F4,  8, (in)[14],  4); \
		ROUND1(E, A, B, C, D, F4,  6, (in)[ 5],  4); \
		ROUND1(D, E, A, B, C, F4,  5, (in)[ 6],  4); \
		ROUND1(C, D, E, A, B, F4, 12, (in)[ 2],  4); \
 \
		ROUND1(B, C, D, E, A, F5,  9, (in)[ 4],  5); \
		ROUND1(A, B, C, D, E, F5, 15, (in)[ 0],  5); \
		ROUND1(E, A, B, C, D, F5,  5, (in)[ 5],  5); \
		ROUND1(D, E, A, B, C, F5, 11, (in)[ 9],  5); \
		ROUND1(C, D, E, A, B, F5,  6, (in)[ 7],  5); \
		ROUND1(B, C, D, E, A, F5,  8, (in)[12],  5); \
		ROUND1(A, B, C, D, E, F5, 13, (in)[ 2],  5); \
		ROUND1(E, A, B, C, D, F5, 12, (in)[10],  5); \
		ROUND1(D, E, A, B, C, F5,  5, (in)[14],  5); \
		ROUND1(C, D, E, A, B, F5, 12, (in)[ 1],  5); \
		ROUND1(B, C, D, E, A, F5, 13, (in)[ 3],  5); \
		ROUND1(A, B, C, D, E, F5, 14, (in)[ 8],  5); \
		ROUND1(E, A, B, C, D, F5, 11, (in)[11],  5); \
		ROUND1(D, E, A, B, C, F5,  8, (in)[ 6],  5); \
		ROUND1(C, D, E, A, B, F5,  5, (in)[15],  5); \
		ROUND1(B, C, D, E, A, F5,  6, (in)[13],  5); \
 \
		ROUND2(A, B, C, D, E, F5,  8, (in)[ 5],  1); \
		ROUND2(E, A, B, C, D, F5,  9, (in)[14],  1); \
		ROUND2(D, E, A, B, C, F5,  9, (in)[ 7],  1); \
		ROUND2(C, D, E, A, B, F5, 11, (in)[ 0],  1); \
		ROUND2(B, C, D, E, A, F5, 13, (in)[ 9],  1); \
		ROUND2(A, B, C, D, E, F5, 15, (in)[ 2],  1); \
		ROUND2(E, A, B, C, D, F5, 15, (in)[11],  1); \
		ROUND2(D, E, A, B, C, F5,  5, (in)[ 4],  1); \
		ROUND2(C, D, E, A, B, F5,  7, (in)[13],  1); \
		ROUND2(B, C, D, E, A, F5,  7, (in)[ 6],  1); \
		ROUND2(A, B, C, D, E, F5,  8, (in)[15],  1); \
		ROUND2(E, A, B, C, D, F5, 11, (in)[ 8],  1); \
		ROUND2(D, E, A, B, C, F5, 14, (in)[ 1],  1); \
		ROUND2(C, D, E, A, B, F5, 14, (in)[10],  1); \
		ROUND2(B, C, D, E, A, F5, 12, (in)[ 3],  1); \
		ROUND2(A, B, C, D, E, F5,  6, (in)[12],  1); \
 \
		ROUND2(E, A, B, C, D, F4,  9, (in)[ 6],  2); \
		ROUND2(D, E, A, B, C, F4, 13, (in)[11],  2); \
		ROUND2(C, D, E, A, B, F4, 15, (in)[ 3],  2); \
		ROUND2(B, C, D, E, A, F4,  7, (in)[ 7],  2); \
		ROUND2(A, B, C, D, E, F4, 12, (in)[ 0],  2); \
		ROUND2(E, A, B, C, D, F4,  8, (in)[13],  2); \
		ROUND2(D, E, A, B, C, F4,  9, (in)[ 5],  2); \
		ROUND2(C, D, E, A, B, F4, 11, (in)[10],  2); \
		ROUND2(B, C, D, E, A, F4,  7, (in)[14],  2); \
		ROUND2(A, B, C, D, E, F4,  7, (in)[15],  2); \
		ROUND2(E, A, B, C, D, F4, 12, (in)[ 8],  2); \
		ROUND2(D, E, A, B, C, F4,  7, (in)[12],  2); \
		ROUND2(C, D, E, A, B, F4,  6, (in)[ 4],  2); \
		ROUND2(B, C, D, E, A, F4, 15, (in)[ 9],  2); \
		ROUND2(A, B, C, D, E, F4, 13, (in)[ 1],  2); \
		ROUND2(E, A, B, C, D, F4, 11, (in)[ 2],  2); \
 \
		ROUND2(D, E, A, B, C, F3,  9, (in)[15],  3); \
		ROUND2(C, D, E, A, B, F3,  7, (in)[ 5],  3); \
		ROUND2(B, C, D, E, A, F3, 15, (in)[ 1],  3); \
		ROUND2(A, B, C, D, E, F3, 11, (in)[ 3],  3); \
		ROUND2(E, A, B, C, D, F3,  8, (in)[ 7],  3); \
		ROUND2(D, E, A, B, C, F3,  6, (in)[14],  3); \
		ROUND2(C, D, E, A, B, F3,  6, (in)[ 6],  3); \
		ROUND2(B, C, D, E, A, F3, 14, (in)[ 9],  3); \
		ROUND2(A, B, C, D, E, F3, 12, (in)[11],  3); \
		ROUND2(E, A, B, C, D, F3, 13, (in)[ 8],  3); \
		ROUND2(D, E, A, B, C, F3,  5, (in)[12],  3); \
		ROUND2(C, D, E, A, B, F3, 14, (in)[ 2],  3); \
		ROUND2(B, C, D, E, A, F3, 13, (in)[10],  3); \
		ROUND2(A, B, C, D, E, F3, 13, (in)[ 0],  3); \
		ROUND2(E, A, B, C, D, F3,  7, (in)[ 4],  3); \
		ROUND2(D, E, A, B, C, F3,  5, (in)[13],  3); \
 \
		ROUND2(C, D, E, A, B, F2, 15, (in)[ 8],  4); \
		ROUND2(B, C, D, E, A, F2,  5, (in)[ 6],  4); \
		ROUND2(A, B, C, D, E, F2,  8, (in)[ 4],  4); \
		ROUND2(E, A, B, C, D, F2, 11, (in)[ 1],  4); \
		ROUND2(D, E, A, B, C, F2, 14, (in)[ 3],  4); \
		ROUND2(C, D, E, A, B, F2, 14, (in)[11],  4); \
		ROUND2(B, C, D, E, A, F2,  6, (in)[15],  4); \
		ROUND2(A, B, C, D, E, F2, 14, (in)[ 0],  4); \
		ROUND2(E, A, B, C, D, F2,  6, (in)[ 5],  4); \
		ROUND2(D, E, A, B, C, F2,  9, (in)[12],  4); \
		ROUND2(C, D, E, A, B, F2, 12, (in)[ 2],  4); \
		ROUND2(B, C, D, E, A, F2,  9, (in)[13],  4); \
		ROUND2(A, B, C, D, E, F2, 12, (in)[ 9],  4); \
		ROUND2(E, A, B, C, D, F2,  5, (in)[ 7],  4); \
		ROUND2(D, E, A, B, C, F2, 15, (in)[10],  4); \
		ROUND2(C, D, E, A, B, F2,  8, (in)[14],  4); \
 \
		ROUND2(B, C, D, E, A, F1,  8, (in)[12],  5); \
		ROUND2(A, B, C, D, E, F1,  5, (in)[15],  5); \
		ROUND2(E, A, B, C, D, F1, 12, (in)[10],  5); \
		ROUND2(D, E, A, B, C, F1,  9, (in)[ 4],  5); \
		ROUND2(C, D, E, A, B, F1, 12, (in)[ 1],  5); \
		ROUND2(B, C, D, E, A, F1,  5, (in)[ 5],  5); \
		ROUND2(A, B, C, D, E, F1, 14, (in)[ 8],  5); \
		ROUND2(E, A, B, C, D, F1,  6, (in)[ 7],  5); \
		ROUND2(D, E, A, B, C, F1,  8, (in)[ 6],  5); \
		ROUND2(C, D, E, A, B, F1, 13, (in)[ 2],  5); \
		ROUND2(B, C, D, E, A, F1,  6, (in)[13],  5); \
		ROUND2(A, B, C, D, E, F1,  5, (in)[14],  5); \
		ROUND2(E, A, B, C, D, F1, 15, (in)[ 0],  5); \
		ROUND2(D, E, A, B, C, F1, 13, (in)[ 3],  5); \
		ROUND2(C, D, E, A, B, F1, 11, (in)[ 9],  5); \
		ROUND2(B, C, D, E, A, F1, 11, (in)[11],  5); \
 \
		tmp = (h)[1] + C1 + D2; \
		(h)[1] = (h)[2] + D1 + E2; \
		(h)[2] = (h)[3] + E1 + A2; \
		(h)[3] = (h)[4] + A1 + B2; \
		(h)[4] = (h)[0] + B1 + C2; \
		(h)[0] = tmp; \
	} while (0)

void __device__ __forceinline__ RIPEMD160_ROUND_BODY(uint32_t *in, uint32_t *h)
{
	uint32_t A1, B1, C1, D1, E1;
	uint32_t A2, B2, C2, D2, E2;
	uint32_t tmp;

	A1 = A2 = (h)[0];
	B1 = B2 = (h)[1];
	C1 = C2 = (h)[2];
	D1 = D2 = (h)[3];
	E1 = E2 = (h)[4];

	ROUND1(A, B, C, D, E, F1, 11, (in)[0], 1);
	ROUND1(E, A, B, C, D, F1, 14, (in)[1], 1);
	ROUND1(D, E, A, B, C, F1, 15, (in)[2], 1);
	ROUND1(C, D, E, A, B, F1, 12, (in)[3], 1);
	ROUND1(B, C, D, E, A, F1, 5, (in)[4], 1);
	ROUND1(A, B, C, D, E, F1, 8, (in)[5], 1);
	ROUND1(E, A, B, C, D, F1, 7, (in)[6], 1);
	ROUND1(D, E, A, B, C, F1, 9, (in)[7], 1);
	ROUND1(C, D, E, A, B, F1, 11, (in)[8], 1);
	ROUND1(B, C, D, E, A, F1, 13, (in)[9], 1);
	ROUND1(A, B, C, D, E, F1, 14, (in)[10], 1);
	ROUND1(E, A, B, C, D, F1, 15, (in)[11], 1);
	ROUND1(D, E, A, B, C, F1, 6, (in)[12], 1);
	ROUND1(C, D, E, A, B, F1, 7, (in)[13], 1);
	ROUND1(B, C, D, E, A, F1, 9, (in)[14], 1);
	ROUND1(A, B, C, D, E, F1, 8, (in)[15], 1);

	ROUND1(E, A, B, C, D, F2, 7, (in)[7], 2);
	ROUND1(D, E, A, B, C, F2, 6, (in)[4], 2);
	ROUND1(C, D, E, A, B, F2, 8, (in)[13], 2);
	ROUND1(B, C, D, E, A, F2, 13, (in)[1], 2);
	ROUND1(A, B, C, D, E, F2, 11, (in)[10], 2);
	ROUND1(E, A, B, C, D, F2, 9, (in)[6], 2);
	ROUND1(D, E, A, B, C, F2, 7, (in)[15], 2);
	ROUND1(C, D, E, A, B, F2, 15, (in)[3], 2);
	ROUND1(B, C, D, E, A, F2, 7, (in)[12], 2);
	ROUND1(A, B, C, D, E, F2, 12, (in)[0], 2);
	ROUND1(E, A, B, C, D, F2, 15, (in)[9], 2);
	ROUND1(D, E, A, B, C, F2, 9, (in)[5], 2);
	ROUND1(C, D, E, A, B, F2, 11, (in)[2], 2);
	ROUND1(B, C, D, E, A, F2, 7, (in)[14], 2);
	ROUND1(A, B, C, D, E, F2, 13, (in)[11], 2);
	ROUND1(E, A, B, C, D, F2, 12, (in)[8], 2);

	ROUND1(D, E, A, B, C, F3, 11, (in)[3], 3);
	ROUND1(C, D, E, A, B, F3, 13, (in)[10], 3);
	ROUND1(B, C, D, E, A, F3, 6, (in)[14], 3);
	ROUND1(A, B, C, D, E, F3, 7, (in)[4], 3);
	ROUND1(E, A, B, C, D, F3, 14, (in)[9], 3);
	ROUND1(D, E, A, B, C, F3, 9, (in)[15], 3);
	ROUND1(C, D, E, A, B, F3, 13, (in)[8], 3);
	ROUND1(B, C, D, E, A, F3, 15, (in)[1], 3);
	ROUND1(A, B, C, D, E, F3, 14, (in)[2], 3);
	ROUND1(E, A, B, C, D, F3, 8, (in)[7], 3);
	ROUND1(D, E, A, B, C, F3, 13, (in)[0], 3);
	ROUND1(C, D, E, A, B, F3, 6, (in)[6], 3);
	ROUND1(B, C, D, E, A, F3, 5, (in)[13], 3);
	ROUND1(A, B, C, D, E, F3, 12, (in)[11], 3);
	ROUND1(E, A, B, C, D, F3, 7, (in)[5], 3);
	ROUND1(D, E, A, B, C, F3, 5, (in)[12], 3);

	ROUND1(C, D, E, A, B, F4, 11, (in)[1], 4);
	ROUND1(B, C, D, E, A, F4, 12, (in)[9], 4);
	ROUND1(A, B, C, D, E, F4, 14, (in)[11], 4);
	ROUND1(E, A, B, C, D, F4, 15, (in)[10], 4);
	ROUND1(D, E, A, B, C, F4, 14, (in)[0], 4);
	ROUND1(C, D, E, A, B, F4, 15, (in)[8], 4);
	ROUND1(B, C, D, E, A, F4, 9, (in)[12], 4);
	ROUND1(A, B, C, D, E, F4, 8, (in)[4], 4);
	ROUND1(E, A, B, C, D, F4, 9, (in)[13], 4);
	ROUND1(D, E, A, B, C, F4, 14, (in)[3], 4);
	ROUND1(C, D, E, A, B, F4, 5, (in)[7], 4);
	ROUND1(B, C, D, E, A, F4, 6, (in)[15], 4);
	ROUND1(A, B, C, D, E, F4, 8, (in)[14], 4);
	ROUND1(E, A, B, C, D, F4, 6, (in)[5], 4);
	ROUND1(D, E, A, B, C, F4, 5, (in)[6], 4);
	ROUND1(C, D, E, A, B, F4, 12, (in)[2], 4);

	ROUND1(B, C, D, E, A, F5, 9, (in)[4], 5);
	ROUND1(A, B, C, D, E, F5, 15, (in)[0], 5);
	ROUND1(E, A, B, C, D, F5, 5, (in)[5], 5);
	ROUND1(D, E, A, B, C, F5, 11, (in)[9], 5);
	ROUND1(C, D, E, A, B, F5, 6, (in)[7], 5);
	ROUND1(B, C, D, E, A, F5, 8, (in)[12], 5);
	ROUND1(A, B, C, D, E, F5, 13, (in)[2], 5);
	ROUND1(E, A, B, C, D, F5, 12, (in)[10], 5);
	ROUND1(D, E, A, B, C, F5, 5, (in)[14], 5);
	ROUND1(C, D, E, A, B, F5, 12, (in)[1], 5);
	ROUND1(B, C, D, E, A, F5, 13, (in)[3], 5);
	ROUND1(A, B, C, D, E, F5, 14, (in)[8], 5);
	ROUND1(E, A, B, C, D, F5, 11, (in)[11], 5);
	ROUND1(D, E, A, B, C, F5, 8, (in)[6], 5);
	ROUND1(C, D, E, A, B, F5, 5, (in)[15], 5);
	ROUND1(B, C, D, E, A, F5, 6, (in)[13], 5);

	ROUND2(A, B, C, D, E, F5, 8, (in)[5], 1);
	ROUND2(E, A, B, C, D, F5, 9, (in)[14], 1);
	ROUND2(D, E, A, B, C, F5, 9, (in)[7], 1);
	ROUND2(C, D, E, A, B, F5, 11, (in)[0], 1);
	ROUND2(B, C, D, E, A, F5, 13, (in)[9], 1);
	ROUND2(A, B, C, D, E, F5, 15, (in)[2], 1);
	ROUND2(E, A, B, C, D, F5, 15, (in)[11], 1);
	ROUND2(D, E, A, B, C, F5, 5, (in)[4], 1);
	ROUND2(C, D, E, A, B, F5, 7, (in)[13], 1);
	ROUND2(B, C, D, E, A, F5, 7, (in)[6], 1);
	ROUND2(A, B, C, D, E, F5, 8, (in)[15], 1);
	ROUND2(E, A, B, C, D, F5, 11, (in)[8], 1);
	ROUND2(D, E, A, B, C, F5, 14, (in)[1], 1);
	ROUND2(C, D, E, A, B, F5, 14, (in)[10], 1);
	ROUND2(B, C, D, E, A, F5, 12, (in)[3], 1);
	ROUND2(A, B, C, D, E, F5, 6, (in)[12], 1);

	ROUND2(E, A, B, C, D, F4, 9, (in)[6], 2);
	ROUND2(D, E, A, B, C, F4, 13, (in)[11], 2);
	ROUND2(C, D, E, A, B, F4, 15, (in)[3], 2);
	ROUND2(B, C, D, E, A, F4, 7, (in)[7], 2);
	ROUND2(A, B, C, D, E, F4, 12, (in)[0], 2);
	ROUND2(E, A, B, C, D, F4, 8, (in)[13], 2);
	ROUND2(D, E, A, B, C, F4, 9, (in)[5], 2);
	ROUND2(C, D, E, A, B, F4, 11, (in)[10], 2);
	ROUND2(B, C, D, E, A, F4, 7, (in)[14], 2);
	ROUND2(A, B, C, D, E, F4, 7, (in)[15], 2);
	ROUND2(E, A, B, C, D, F4, 12, (in)[8], 2);
	ROUND2(D, E, A, B, C, F4, 7, (in)[12], 2);
	ROUND2(C, D, E, A, B, F4, 6, (in)[4], 2);
	ROUND2(B, C, D, E, A, F4, 15, (in)[9], 2);
	ROUND2(A, B, C, D, E, F4, 13, (in)[1], 2);
	ROUND2(E, A, B, C, D, F4, 11, (in)[2], 2);

	ROUND2(D, E, A, B, C, F3, 9, (in)[15], 3);
	ROUND2(C, D, E, A, B, F3, 7, (in)[5], 3);
	ROUND2(B, C, D, E, A, F3, 15, (in)[1], 3);
	ROUND2(A, B, C, D, E, F3, 11, (in)[3], 3);
	ROUND2(E, A, B, C, D, F3, 8, (in)[7], 3);
	ROUND2(D, E, A, B, C, F3, 6, (in)[14], 3);
	ROUND2(C, D, E, A, B, F3, 6, (in)[6], 3);
	ROUND2(B, C, D, E, A, F3, 14, (in)[9], 3);
	ROUND2(A, B, C, D, E, F3, 12, (in)[11], 3);
	ROUND2(E, A, B, C, D, F3, 13, (in)[8], 3);
	ROUND2(D, E, A, B, C, F3, 5, (in)[12], 3);
	ROUND2(C, D, E, A, B, F3, 14, (in)[2], 3);
	ROUND2(B, C, D, E, A, F3, 13, (in)[10], 3);
	ROUND2(A, B, C, D, E, F3, 13, (in)[0], 3);
	ROUND2(E, A, B, C, D, F3, 7, (in)[4], 3);
	ROUND2(D, E, A, B, C, F3, 5, (in)[13], 3);

	ROUND2(C, D, E, A, B, F2, 15, (in)[8], 4);
	ROUND2(B, C, D, E, A, F2, 5, (in)[6], 4);
	ROUND2(A, B, C, D, E, F2, 8, (in)[4], 4);
	ROUND2(E, A, B, C, D, F2, 11, (in)[1], 4);
	ROUND2(D, E, A, B, C, F2, 14, (in)[3], 4);
	ROUND2(C, D, E, A, B, F2, 14, (in)[11], 4);
	ROUND2(B, C, D, E, A, F2, 6, (in)[15], 4);
	ROUND2(A, B, C, D, E, F2, 14, (in)[0], 4);
	ROUND2(E, A, B, C, D, F2, 6, (in)[5], 4);
	ROUND2(D, E, A, B, C, F2, 9, (in)[12], 4);
	ROUND2(C, D, E, A, B, F2, 12, (in)[2], 4);
	ROUND2(B, C, D, E, A, F2, 9, (in)[13], 4);
	ROUND2(A, B, C, D, E, F2, 12, (in)[9], 4);
	ROUND2(E, A, B, C, D, F2, 5, (in)[7], 4);
	ROUND2(D, E, A, B, C, F2, 15, (in)[10], 4);
	ROUND2(C, D, E, A, B, F2, 8, (in)[14], 4);

	ROUND2(B, C, D, E, A, F1, 8, (in)[12], 5);
	ROUND2(A, B, C, D, E, F1, 5, (in)[15], 5);
	ROUND2(E, A, B, C, D, F1, 12, (in)[10], 5);
	ROUND2(D, E, A, B, C, F1, 9, (in)[4], 5);
	ROUND2(C, D, E, A, B, F1, 12, (in)[1], 5);
	ROUND2(B, C, D, E, A, F1, 5, (in)[5], 5);
	ROUND2(A, B, C, D, E, F1, 14, (in)[8], 5);
	ROUND2(E, A, B, C, D, F1, 6, (in)[7], 5);
	ROUND2(D, E, A, B, C, F1, 8, (in)[6], 5);
	ROUND2(C, D, E, A, B, F1, 13, (in)[2], 5);
	ROUND2(B, C, D, E, A, F1, 6, (in)[13], 5);
	ROUND2(A, B, C, D, E, F1, 5, (in)[14], 5);
	ROUND2(E, A, B, C, D, F1, 15, (in)[0], 5);
	ROUND2(D, E, A, B, C, F1, 13, (in)[3], 5);
	ROUND2(C, D, E, A, B, F1, 11, (in)[9], 5);
	ROUND2(B, C, D, E, A, F1, 11, (in)[11], 5);

	tmp = (h)[1] + C1 + D2;
	(h)[1] = (h)[2] + D1 + E2;
	(h)[2] = (h)[3] + E1 + A2;
	(h)[3] = (h)[4] + A1 + B2;
	(h)[4] = (h)[0] + B1 + C2;
	(h)[0] = tmp;
}


#define ROL32(x, y)		ROTL32(x,y) //rotate(x, y ## U)
#define SHR(x, y)		(x >> y)
//#define SWAP32(a)    	(as_uint(as_uchar4(a).wzyx))

#define S0(x) (ROL32(x, 25) ^ ROL32(x, 14) ^  SHR(x, 3))
#define S1(x) (ROL32(x, 15) ^ ROL32(x, 13) ^  SHR(x, 10))

#define S2(x) (ROL32(x, 30) ^ ROL32(x, 19) ^ ROL32(x, 10))
#define S3(x) (ROL32(x, 26) ^ ROL32(x, 21) ^ ROL32(x, 7))

#define P(a,b,c,d,e,f,g,h,x,K)                  \
{                                               \
    temp1 = h + S3(e) + F1(e,f,g) + (K + x);      \
    d += temp1; h = temp1 + S2(a) + F0(a,b,c);  \
}

#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))


#define F0(y, x, z) bitselect(z, y, z ^ x)
//#define F1(x, y, z) bitselect(z, y, x)

#define R0 (W0 = S1(W14) + W9 + S0(W1) + W0)
#define R1 (W1 = S1(W15) + W10 + S0(W2) + W1)
#define R2 (W2 = S1(W0) + W11 + S0(W3) + W2)
#define R3 (W3 = S1(W1) + W12 + S0(W4) + W3)
#define R4 (W4 = S1(W2) + W13 + S0(W5) + W4)
#define R5 (W5 = S1(W3) + W14 + S0(W6) + W5)
#define R6 (W6 = S1(W4) + W15 + S0(W7) + W6)
#define R7 (W7 = S1(W5) + W0 + S0(W8) + W7)
#define R8 (W8 = S1(W6) + W1 + S0(W9) + W8)
#define R9 (W9 = S1(W7) + W2 + S0(W10) + W9)
#define R10 (W10 = S1(W8) + W3 + S0(W11) + W10)
#define R11 (W11 = S1(W9) + W4 + S0(W12) + W11)
#define R12 (W12 = S1(W10) + W5 + S0(W13) + W12)
#define R13 (W13 = S1(W11) + W6 + S0(W14) + W13)
#define R14 (W14 = S1(W12) + W7 + S0(W15) + W14)
#define R15 (W15 = S1(W13) + W8 + S0(W0) + W15)

#define RD14 (S1(W12) + W7 + S0(W15) + W14)
#define RD15 (S1(W13) + W8 + S0(W0) + W15)

struct uint8
{
	uint32_t s0;
	uint32_t s1;
	uint32_t s2;
	uint32_t s3;
	uint32_t s4;
	uint32_t s5;
	uint32_t s6;
	uint32_t s7;
};

struct uint16
{
	uint32_t s0;
	uint32_t s1;
	uint32_t s2;
	uint32_t s3;
	uint32_t s4;
	uint32_t s5;
	uint32_t s6;
	uint32_t s7;
	uint32_t s8;
	uint32_t s9;
	uint32_t sA;
	uint32_t sB;
	uint32_t sC;
	uint32_t sD;
	uint32_t sE;
	uint32_t sF;
};


__device__ __forceinline__ uint8 sha256_round(uint16 data, uint8 buf)
{
	uint32_t temp1;
	uint8 res;
	uint32_t W0 = (data.s0);
	uint32_t W1 = (data.s1);
	uint32_t W2 = (data.s2);
	uint32_t W3 = (data.s3);
	uint32_t W4 = (data.s4);
	uint32_t W5 = (data.s5);
	uint32_t W6 = (data.s6);
	uint32_t W7 = (data.s7);
	uint32_t W8 = (data.s8);
	uint32_t W9 = (data.s9);
	uint32_t W10 = (data.sA);
	uint32_t W11 = (data.sB);
	uint32_t W12 = (data.sC);
	uint32_t W13 = (data.sD);
	uint32_t W14 = (data.sE);
	uint32_t W15 = (data.sF);

	uint32_t v0 = buf.s0;
	uint32_t v1 = buf.s1;
	uint32_t v2 = buf.s2;
	uint32_t v3 = buf.s3;
	uint32_t v4 = buf.s4;
	uint32_t v5 = buf.s5;
	uint32_t v6 = buf.s6;
	uint32_t v7 = buf.s7;

	P(v0, v1, v2, v3, v4, v5, v6, v7, W0, 0x428A2F98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, W1, 0x71374491);
	P(v6, v7, v0, v1, v2, v3, v4, v5, W2, 0xB5C0FBCF);
	P(v5, v6, v7, v0, v1, v2, v3, v4, W3, 0xE9B5DBA5);
	P(v4, v5, v6, v7, v0, v1, v2, v3, W4, 0x3956C25B);
	P(v3, v4, v5, v6, v7, v0, v1, v2, W5, 0x59F111F1);
	P(v2, v3, v4, v5, v6, v7, v0, v1, W6, 0x923F82A4);
	P(v1, v2, v3, v4, v5, v6, v7, v0, W7, 0xAB1C5ED5);
	P(v0, v1, v2, v3, v4, v5, v6, v7, W8, 0xD807AA98);
	P(v7, v0, v1, v2, v3, v4, v5, v6, W9, 0x12835B01);
	P(v6, v7, v0, v1, v2, v3, v4, v5, W10, 0x243185BE);
	P(v5, v6, v7, v0, v1, v2, v3, v4, W11, 0x550C7DC3);
	P(v4, v5, v6, v7, v0, v1, v2, v3, W12, 0x72BE5D74);
	P(v3, v4, v5, v6, v7, v0, v1, v2, W13, 0x80DEB1FE);
	P(v2, v3, v4, v5, v6, v7, v0, v1, W14, 0x9BDC06A7);
	P(v1, v2, v3, v4, v5, v6, v7, v0, W15, 0xC19BF174);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0xE49B69C1);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0xEFBE4786);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x0FC19DC6);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x240CA1CC);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x2DE92C6F);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x4A7484AA);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x5CB0A9DC);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x76F988DA);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0x983E5152);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0xA831C66D);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0xB00327C8);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0xBF597FC7);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0xC6E00BF3);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xD5A79147);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R14, 0x06CA6351);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R15, 0x14292967);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0x27B70A85);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0x2E1B2138);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x4D2C6DFC);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x53380D13);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x650A7354);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x766A0ABB);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x81C2C92E);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x92722C85);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0xA2BFE8A1);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0xA81A664B);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0xC24B8B70);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0xC76C51A3);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0xD192E819);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xD6990624);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R14, 0xF40E3585);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R15, 0x106AA070);

	P(v0, v1, v2, v3, v4, v5, v6, v7, R0, 0x19A4C116);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R1, 0x1E376C08);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R2, 0x2748774C);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R3, 0x34B0BCB5);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R4, 0x391C0CB3);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R5, 0x4ED8AA4A);
	P(v2, v3, v4, v5, v6, v7, v0, v1, R6, 0x5B9CCA4F);
	P(v1, v2, v3, v4, v5, v6, v7, v0, R7, 0x682E6FF3);
	P(v0, v1, v2, v3, v4, v5, v6, v7, R8, 0x748F82EE);
	P(v7, v0, v1, v2, v3, v4, v5, v6, R9, 0x78A5636F);
	P(v6, v7, v0, v1, v2, v3, v4, v5, R10, 0x84C87814);
	P(v5, v6, v7, v0, v1, v2, v3, v4, R11, 0x8CC70208);
	P(v4, v5, v6, v7, v0, v1, v2, v3, R12, 0x90BEFFFA);
	P(v3, v4, v5, v6, v7, v0, v1, v2, R13, 0xA4506CEB);
	P(v2, v3, v4, v5, v6, v7, v0, v1, RD14, 0xBEF9A3F7);
	P(v1, v2, v3, v4, v5, v6, v7, v0, RD15, 0xC67178F2);

	res.s0 = (v0 + buf.s0);
	res.s1 = (v1 + buf.s1);
	res.s2 = (v2 + buf.s2);
	res.s3 = (v3 + buf.s3);
	res.s4 = (v4 + buf.s4);
	res.s5 = (v5 + buf.s5);
	res.s6 = (v6 + buf.s6);
	res.s7 = (v7 + buf.s7);
	return (res);
}

__global__ void search(uint32_t threads, uint32_t startNounce, uint8 *ctx)
{
	// SHA256 takes 16 uints of input per block - we have 112 bytes to process
	// 8 * 16 == 64, meaning two block transforms.

	uint32_t SHA256Buf[16];

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (startNounce + thread);
		uint32_t hashPosition = (nounce - startNounce);

//		uint32_t gid = 1;// get_global_id(0);

		// Remember the last four is the nonce - so 108 bytes / 4 bytes per dword
#pragma unroll
		for (int i = 0; i < 16; ++i) SHA256Buf[i] = cuda_swab32(c_data[i]);



		// SHA256 initialization constants
		//	uint8 outbuf; = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);
		uint8 outbuf;
		outbuf.s0 = 0x6A09E667;
		outbuf.s1 = 0xBB67AE85;
		outbuf.s2 = 0x3C6EF372;
		outbuf.s3 = 0xA54FF53A;
		outbuf.s4 = 0x510E527F;
		outbuf.s5 = 0x9B05688C;
		outbuf.s6 = 0x1F83D9AB;
		outbuf.s7 = 0x5BE0CD19;

#pragma unroll
		for (int i = 0; i < 3; ++i)
		{
			if (i == 1)
			{
#pragma unroll
				for (int i = 0; i < 11; ++i) SHA256Buf[i] = cuda_swab32(c_data[i + 16]);
				SHA256Buf[11] = cuda_swab32(hashPosition);
				SHA256Buf[12] = 0x80000000;
				SHA256Buf[13] = 0x00000000;
				SHA256Buf[14] = 0x00000000;
				SHA256Buf[15] = 0x00000380;
			}
			if (i == 2)
			{
				((uint8 *)SHA256Buf)[0] = outbuf;
				SHA256Buf[8] = 0x80000000;
#pragma unroll
				for (int i = 9; i < 15; ++i) SHA256Buf[i] = 0x00000000;
				SHA256Buf[15] = 0x00000100;
//				outbuf = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);
				outbuf.s0 = 0x6A09E667;
				outbuf.s1 = 0xBB67AE85;
				outbuf.s2 = 0x3C6EF372;
				outbuf.s3 = 0xA54FF53A;
				outbuf.s4 = 0x510E527F;
				outbuf.s5 = 0x9B05688C;
				outbuf.s6 = 0x1F83D9AB;
				outbuf.s7 = 0x5BE0CD19;
			}
			outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		}

		/*
		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		#pragma unroll
		for(int i = 0; i < 11; ++i) SHA256Buf[i] = SWAP32(input[i + 16]);
		SHA256Buf[11] = SWAP32(gid);
		SHA256Buf[12] = 0x80000000;
		SHA256Buf[13] = 0x00000000;
		SHA256Buf[14] = 0x00000000;
		SHA256Buf[15] = 0x00000380;

		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		((uint8 *)SHA256Buf)[0] = outbuf;
		SHA256Buf[8] = 0x80000000;
		for(int i = 9; i < 15; ++i) SHA256Buf[i] = 0x00000000;
		SHA256Buf[15] = 0x00000100;
		outbuf = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);
		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		*/


		/*

		//outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		//outbuf = sha256_round(((uint16 *)SHA256Buf)[1], outbuf);

		// outbuf would normall be SWAP32'd here, but it'll need it again
		// once we use it as input to the next SHA256, so it negates.

		((uint8 *)SHA256Buf)[0] = outbuf;
		SHA256Buf[8] = 0x80000000;
		for(int i = 9; i < 15; ++i) SHA256Buf[i] = 0x00000000;
		SHA256Buf[15] = 0x00000100;

		outbuf = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);
		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		*/


		/*
		outbuf.s0 = cuda_swab32(outbuf.s0);
		outbuf.s1 = cuda_swab32(outbuf.s1);
		outbuf.s2 = cuda_swab32(outbuf.s2);
		outbuf.s3 = cuda_swab32(outbuf.s3);
		outbuf.s4 = cuda_swab32(outbuf.s4);
		outbuf.s5 = cuda_swab32(outbuf.s5);
		outbuf.s6 = cuda_swab32(outbuf.s6);
		outbuf.s7 = cuda_swab32(outbuf.s7);

		ctx[hashPosition] = outbuf;
		*/


		//	ctx[get_global_id(0) - get_global_offset(0)] = outbuf;
	}
}

__global__ void search1(uint32_t threads, uint32_t startNounce, uint8 *ctx)
{
	uint64_t W[16] = { 0UL }, SHA512Out[8];
	uint32_t SHA256Buf[16];

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (startNounce + thread);
		uint32_t hashPosition = (nounce - startNounce);

		uint8 outbuf = ctx[hashPosition];  //[get_global_id(0) - get_global_offset(0)];

		((uint8 *)W)[0] = outbuf;

		for (int i = 0; i < 4; ++i) W[i] = SWAP64(W[i]);

		W[4] = 0x8000000000000000UL;
		W[15] = 0x0000000000000100UL;

		for (int i = 0; i < 8; ++i) SHA512Out[i] = SHA512_INIT[i];

		SHA512Block(W, SHA512Out);

		for (int i = 0; i < 8; ++i) SHA512Out[i] = SWAP64(SHA512Out[i]);

		uint32_t RMD160_0[16] = { 0U };
		uint32_t RMD160_1[16] = { 0U };
		uint32_t RMD160_0_Out[5], RMD160_1_Out[5];

		for (int i = 0; i < 4; ++i)
		{
			((uint64_t *)RMD160_0)[i] = SHA512Out[i];
			((uint64_t *)RMD160_1)[i] = SHA512Out[i + 4];
		}

		RMD160_0[8] = RMD160_1[8] = 0x00000080;
		RMD160_0[14] = RMD160_1[14] = 0x00000100;

		for (int i = 0; i < 5; ++i)
		{
			RMD160_0_Out[i] = RMD160_IV[i];
			RMD160_1_Out[i] = RMD160_IV[i];
		}

		RIPEMD160_ROUND_BODY(RMD160_0, RMD160_0_Out);
		RIPEMD160_ROUND_BODY(RMD160_1, RMD160_1_Out);

		for (int i = 0; i < 5; ++i) SHA256Buf[i] = cuda_swab32(RMD160_0_Out[i]);
		for (int i = 5; i < 10; ++i) SHA256Buf[i] = cuda_swab32(RMD160_1_Out[i - 5]);
		SHA256Buf[10] = 0x80000000;

		for (int i = 11; i < 15; ++i) SHA256Buf[i] = 0x00000000U;

		SHA256Buf[15] = 0x00000140;

		//	outbuf = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);
		outbuf.s0 = 0x6A09E667;
		outbuf.s1 = 0xBB67AE85;
		outbuf.s2 = 0x3C6EF372;
		outbuf.s3 = 0xA54FF53A;
		outbuf.s4 = 0x510E527F;
		outbuf.s5 = 0x9B05688C;
		outbuf.s6 = 0x1F83D9AB;
		outbuf.s7 = 0x5BE0CD19;


		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);
		ctx[hashPosition] = outbuf;
	}


}

__global__ void search2(uint32_t threads, uint32_t startNounce, uint8 *ctx, uint32_t *d_found)
{

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (startNounce + thread);
		uint32_t hashPosition = (nounce - startNounce);
		uint32_t SHA256Buf[16] = { 0U };
		uint8 outbuf = ctx[hashPosition];//get_global_id(0) - get_global_offset(0)];

		((uint8 *)SHA256Buf)[0] = outbuf;
		SHA256Buf[8] = 0x80000000;
		for (int i = 9; i < 15; ++i) SHA256Buf[i] = 0x00000000;
		SHA256Buf[15] = 0x00000100;

		//	outbuf = (uint8)(0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19);

		outbuf.s0 = 0x6A09E667;
		outbuf.s1 = 0xBB67AE85;
		outbuf.s2 = 0x3C6EF372;
		outbuf.s3 = 0xA54FF53A;
		outbuf.s4 = 0x510E527F;
		outbuf.s5 = 0x9B05688C;
		outbuf.s6 = 0x1F83D9AB;
		outbuf.s7 = 0x5BE0CD19;

		outbuf = sha256_round(((uint16 *)SHA256Buf)[0], outbuf);

		outbuf.s6 = cuda_swab32(outbuf.s6);
		outbuf.s7 = cuda_swab32(outbuf.s7);

		uint64_t test = MAKE_ULONGLONG(outbuf.s7, outbuf.s6);
		//if(!(outbuf.s7)) output[atomic_inc(output+0xFF)] = SWAP32(gid);	
		if (test <= ((uint64_t *)pTarget)[3])
		{
			//yai.
			uint32_t tmp = atomicCAS(d_found, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				d_found[1] = nounce;

		}
		//		output[atomic_inc(output + 0xFF)] = SWAP32(gid);
	}
}


__host__ void lbrcredit_cpu_hash(uint32_t thr_id, int threads, uint32_t startNounce, const uint32_t *const __restrict__ g_hash, uint32_t *h_found)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));

	search << <grid, block >> >(threads, startNounce, (uint8*)g_hash);
	search1 << <grid, block >> >(threads, startNounce, (uint8 *)g_hash);
	search2 << <grid, block >> >(threads, startNounce, (uint8 *)g_hash, d_found[thr_id]);

	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

__host__ void lbrcredit_setBlockTarget(uint32_t* pdata, const void *target)
{

	unsigned char PaddedMessage[192];
	memcpy(PaddedMessage, pdata, 168);
	memset(PaddedMessage + 168, 0, 24);
	((uint32_t*)PaddedMessage)[42] = 0x80000000;
	((uint32_t*)PaddedMessage)[47] = 0x0540;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, target, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, PaddedMessage, 48 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

}
