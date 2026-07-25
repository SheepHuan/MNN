# MNNPerfCounter

Optional GPU performance-counter support for `replay_benchmark`.

The legacy runtime implementation is based on Google
[HardwarePerfCounter](https://github.com/google/hardware-perfcounter) at
commit `7949f047e58e44996236e6ec6153cfb60951c37a`. The wrapper automatically
detects Adreno (`/dev/kgsl-3d0`) or Mali (`/dev/mali0`), reads the product ID,
selects the matching counter layout, and exposes normalized counter names
through `MNNPerfCounter`.

Adreno A7xx support is generated from the Mesa snapshot in
`upstream/mesa/freedreno/registers/adreno/`:

- `a7xx_perfcntrs.xml`: event selector/countable definitions;
- `a7xx_perfcntrs.json`: physical slots and register layout;
- `src/A7xxPerfCounters.cpp`: generated 885-event runtime table.

KGSL group IDs are an explicit userspace ABI table in the generator. They are
not inferred from Mesa JSON order. The important A7xx groups are SP=10,
HLSQ=4, UCHE=8, TP=9, RBBM=1 and CP=0.

ARM [libGPUCounters](https://github.com/ARM-software/libGPUCounters) is vendored
under `upstream/libgpucounters/` for complete Mali PMU support. Product ID
decoding covers Midgard, Bifrost, Valhall and Arm 5th Gen; HWCpipe selects the
matching `vinstr`/`kinstr_prfcnt` driver backend and product-specific counter
database at runtime. Its counter database is initialized lazily inside a
function to keep MNN's startup rules intact.

The library is enabled by default for `replay_benchmark` and is linked only into
`replay_benchmark.out`; `-DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF` restores the
small legacy build. It does not change normal MNN binaries.

## Updating the upstream snapshot

Update each upstream snapshot from a reviewed commit, retain all upstream
copyright headers and licenses, and re-run the host and AArch64 replay
validations. Regenerate the A7xx table with:

```bash
python3 tools/generate_a7xx_events.py \
  upstream/mesa/freedreno/registers/adreno/a7xx_perfcntrs.xml \
  upstream/mesa/freedreno/registers/adreno/a7xx_perfcntrs.json \
  src/A7xxPerfCounters.cpp
```

## Normalized counters

The default replay profile contains GPU activity, compute activity/task counts,
and L2 traffic/lookups. Use `--perf-counter-events name1,name2,...` to request
any generated A7xx event or any descriptive Arm HWCpipe counter. A counter
unsupported by the detected product is reported as an explicit
unavailable/error result rather than being read with a guessed layout.

For Rhinopi event coverage, first record an OpenCL operator and run:

```bash
python3 tools/probe_a7xx_replay.py \
  --device root@192.168.101.227 \
  --model models/mobilenetV3.mnn \
  --record records/opencl/mobilenetV3.mnn \
  --op-id 0 \
  --csv rhinopi-a7xx-events.csv
```

The script batches by physical counter slots and reports each event's
configuration status and measured delta. A non-zero delta means the event was
read successfully and changed during the selected operator interval; a zero
delta is a valid result for events not exercised by that operator.
