
extern "C"
{
#include "sph/sph_types.h"
#include "sph/sph_sha2.h"
#include "miner.h"
}

#include "cuda_helper.h"



static uint32_t *d_hash[MAX_GPUS];
static uint32_t foundnonces[MAX_GPUS][2];



extern void bitcredit_setBlockTarget(uint32_t * data,const uint32_t * midstate, const void *ptarget);
extern void bitcredit_cpu_init(uint32_t thr_id, int threads, uint32_t* hash);
//extern uint32_t bitcredit_cpu_hash(uint32_t thr_id, int threads, uint32_t startNounce, int order);
extern void lbrcredit_cpu_hash(uint32_t thr_id, int threads, uint32_t startNounce, const uint32_t *const __restrict__ g_hash, uint32_t *h_found);
extern void lbrcredit_setBlockTarget(uint32_t* pdata, const void *target);


 void credithash(void *state, const void *input)
{

	sph_sha256_context sha1,sha2;
	uint32_t hash[8],hash2[8];

	sph_sha256_init(&sha1);
	sph_sha256(&sha1, input, 168);
	sph_sha256_close(&sha1, hash);


	sph_sha256_init(&sha2);
	sph_sha256(&sha2, hash, 32);
	sph_sha256_close(&sha2, hash2);


	memcpy(state, hash2, 32);

 }


     
extern "C" int scanhash_bitcredit(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, const uint32_t *midstate, uint32_t max_nonce,
	unsigned long *hashes_done)
{


	const uint32_t first_nonce = pdata[35];
	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x1;

	const uint32_t Htarg = ptarget[7];
	int coef = 4;

	uint32_t throughput = 256*256*64*8;

	static bool init[MAX_GPUS] = { 0 };
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]); 
		cudaDeviceReset();
		if (!opt_cpumining) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		 
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id],  8 * sizeof(uint32_t) * throughput));
	//	lbrcredit_cpu_init(thr_id, throughput, d_hash[thr_id]);
		init[thr_id] = true;
	}

	uint32_t endiandata[42],endianmid[8];
		for (int k = 0; k < 42; k++)
			be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	lbrcredit_setBlockTarget(pdata,ptarget);
	uint64_t nloop = max_nonce/throughput + 1;
	do {
		lbrcredit_cpu_hash(thr_id, throughput, pdata[35], d_hash[thr_id], foundnonces[thr_id]);
		if (foundnonces[thr_id][0] != 0xffffffff)
		{
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, foundnonces[thr_id][0]);

				pdata[35] = foundnonces[thr_id][0];
				*hashes_done = foundnonces[thr_id][0] - first_nonce;
				return 1;

		}
		if ((uint64_t)pdata[35] + throughput >(uint64_t)0xffffffff) {
                       pdata[35]=0xffffffff; 
                      *hashes_done = pdata[35] - first_nonce; return 0;
        } else { 
    
		pdata[35] += throughput;}
	} while (!scan_abort_flag && !work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[35] - first_nonce;
	return 0;
}
