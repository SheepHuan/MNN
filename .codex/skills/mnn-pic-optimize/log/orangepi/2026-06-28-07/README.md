# OrangePi 2026-06-28 07

## Summary

- Issue: OrangePi Qwen3-8B `ctx2048 cacheblend 0.50` warm request can hard-hang the device: HTTP returns `curl: (52) Empty reply from server`, SSH then times out during banner exchange, and no `benchmark.csv` row is valid.
- First diagnosis: `prefill_text` was a cache hit and succeeded; failure occurs during cacheblend warm. The server log did not show a C++ exception before the device became unresponsive.
- Route narrowed: forcing Mali `headDim=128` sparse flash away from mqtile/row64 to row32 allowed progress through later sparse layers but did not fully prevent the hang.
- Root cause found: `runSparseFastPrefill()` reused `ensureFastPrefillTemps()`, which allocates static `mTempQK` and `mTempSoftmax` full `[activeLen, kvLen, heads]` buffers per layer. Sparse flash does not use these buffers. At Qwen3-8B `ctx2048/cb50`, this pushes OrangePi memory to roughly full system RAM.
- Fix implemented locally: add `ensureSparseFlashTemps()` and call it from `runSparseFastPrefill()`, allocating only Q/K/V temp buffers for sparse flash. Also keep Mali `headDim=128` sparse flash on row32.
- Status: OrangePi was still SSH-unresponsive after the pre-fix repro, so the memory fix was built locally but not yet synced/tested on device in this hour.

## Files

- `context.md`: commands, observations, and implementation details.
