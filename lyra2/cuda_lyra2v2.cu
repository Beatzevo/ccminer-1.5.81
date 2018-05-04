

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#define TPB52 256
#define TPB50 64

 
#define Nrow 4
#define Ncol 4
#define u64type uint2
#define vectype uint28
#define memshift 3
__device__ vectype  *DMatrix;

 
__device__ __forceinline__ void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{

	a += b; d = eorswap32 (a, d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);

}

__device__ __forceinline__ void round_lyra_v35(vectype* s)
{

	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);

}


 

__device__ __forceinline__ void reduceDuplex50(vectype state[4], uint32_t thread)
{
	const uint32_t ps1 = (Nrow * Ncol * memshift * thread);
	const uint32_t ps2 = (memshift * (Ncol - 1) + memshift * Ncol + Nrow * Ncol * memshift * thread);
	uint28 tmp[3];

//#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
#if __CUDA_ARCH__ == 500

		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 - i*memshift;

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + s1)[j]);

		round_lyra_v35(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = __ldg4(&(DMatrix + s1)[j]) ^ state[j];
#else
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 - i*memshift;
		tmp[0] = __ldg4(&(DMatrix + s1)[0]);
		tmp[1] = __ldg4(&(DMatrix + s1)[1]);
		tmp[2] = __ldg4(&(DMatrix + s1)[2]);
		state[0] ^= tmp[0];
		state[1] ^= tmp[1];
		state[2] ^= tmp[2];

		round_lyra_v35(state);

#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = tmp[j] ^ state[j];
#endif

	}
}
__device__  void reduceDuplexRowSetupV2(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{

	int i, j;
		vectype state2[3],state1[3];

		const uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
		const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
		const uint32_t ps3 = (memshift * (Ncol-1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);
	for (i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 - i*memshift;

		#if __CUDA_ARCH__ == 500
		#pragma unroll
		for (j = 0; j < 3; j++)
		{
			state[j] = state[j] ^ (__ldg4(&(DMatrix + s1)[j]) + __ldg4(&(DMatrix + s2)[j]));
		}
		
		round_lyra_v35(state);
#pragma unroll
		for (j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

#pragma unroll
		for (j = 0; j < 3; j++) 
		{
			(DMatrix + s3)[j] =state[j]^state1[j];
		}
		#else

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);
#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);
#pragma unroll
		for (j = 0; j < 3; j++)
		{
			state[j] ^= state1[j] + state2[j];
		}

		round_lyra_v35(state);

#pragma unroll
		for (j = 0; j < 3; j++)
		{			
			(DMatrix + s3)[j] = state1[j]^ state[j];;
		}

		#endif

		   ((uint2*)state2)[0] ^= ((uint2*)state)[11];
		   #pragma unroll
		   for (j = 0; j < 11; j++)
			((uint2*)state2)[j+1] ^= ((uint2*)state)[j];


		#pragma unroll
		for (j = 0; j < 3; j++)
		    (DMatrix + s2)[j] = state2[j];
	}
}



__device__ void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{
	int i,j;
		vectype state2[3];
		const uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
		const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
		const uint32_t ps3 = (memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);
	
	for (i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 + i*memshift;

		#pragma unroll 
		for (j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

		#pragma unroll 
		for (j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + s1)[j]) + state2[j];

		round_lyra_v35(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		#pragma unroll 
		for (j = 0; j < 11; j++)
		((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];
#if __CUDA_ARCH__ == 500
		if (rowInOut != rowOut) 
		{
			#pragma unroll 
			for ( j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];

		} 
		if (rowInOut == rowOut)
		{
			#pragma unroll 
			for (j = 0; j < 3; j++)
			state2[j] ^= state[j];
		}
#else
		if (rowInOut != rowOut)
		{
			#pragma unroll 
			for (j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];

		} else
		{
			#pragma unroll 
			for (j = 0; j < 3; j++)
				state2[j] ^= state[j];
		}
#endif

		#pragma unroll 
		for (j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 1)
#endif
void lyra2v2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);



	vectype state[4];

	if (thread < threads)
	{
		const uint28 blake2b_IV[2] =
		{
			0xf3bcc908, 0x6a09e667,
			0x84caa73b, 0xbb67ae85,
			0xfe94f82b, 0x3c6ef372,
			0x5f1d36f1, 0xa54ff53a,
			0xade682d1, 0x510e527f,
			0x2b3e6c1f, 0x9b05688c,
			0xfb41bd6b, 0x1f83d9ab,
			0x137e2179, 0x5be0cd19
		};

		state[2] = ((blake2b_IV)[0]);
		state[3] = ((blake2b_IV)[1]);

		((uint2*)state)[0] = __ldg(&outputHash[thread]);
		((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);

		 state[1] = state[0];

		 for (int i = 0; i<12; i++)
			 round_lyra_v35(state);
		 ((uint2*)state)[0].x ^= 0x20;
		 ((uint2*)state)[1].x ^= 0x20;
		 ((uint2*)state)[2].x ^= 0x20;
		 ((uint2*)state)[3].x ^= 0x01;
		 ((uint2*)state)[4].x ^= 0x04;
		 ((uint2*)state)[5].x ^= 0x04;
		 ((uint2*)state)[6].x ^= 0x80;
		 ((uint2*)state)[7].y ^= 0x01000000;

		 for (int i = 0; i<12; i++)
			 round_lyra_v35(state);

		const uint32_t ps1 = (memshift * (Ncol - 1) + Nrow * Ncol * memshift * thread);

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
		for (int i = 0; i < Ncol; i++)
		{
			const uint32_t s1 = ps1 - memshift * i;
			DMatrix[s1] = state[0];
			DMatrix[s1+1] = state[1];
			DMatrix[s1+2] = state[2];
			round_lyra_v35(state);
		}

		reduceDuplex50(state, thread);

		reduceDuplexRowSetupV2(1, 0, 2, state,  thread);
		reduceDuplexRowSetupV2(2, 1, 3, state,  thread);

		uint32_t rowa;
		int prev=3;

        for (int i = 0; i < 4; i++)
        {
	     rowa = ((uint2*)state)[0].x & 3;  
		 reduceDuplexRowtV2(prev, rowa, i, state, thread);
         prev=i;
        }


		const uint32_t shift = (memshift * Ncol * rowa + Nrow * Ncol * memshift * thread);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
        	round_lyra_v35(state);
		
		outputHash[thread] = ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2];
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];
//		((vectype*)outputHash)[thread] = state[0];

	} //thread
}


__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads,uint64_t *hash)
{
	cudaMemcpyToSymbol(DMatrix, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}



__host__ 
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb)
{
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	lyra2v2_gpu_hash_32 << <grid, block >> > (threads, startNounce, (uint2*)d_outputHash);
}

  