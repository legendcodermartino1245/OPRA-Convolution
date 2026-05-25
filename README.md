# FIR DSP Toolchain

A deterministic Python toolchain for building minimum-phase FIR filters from:

- FFT-binned linear magnitude targets
- two-column frequency/response files in linear or dB form
- OPRA parametric EQ entries

This repository is structured like an engineering tool rather than a one-off DSP notebook: validation is centralized, outputs are auditable, the CLI exports verification metadata, and the test suite checks determinism, correctness, and export behavior.

## Signal Flow

```text
Input response or OPRA EQ
  -> centralized validation
  -> exact FFT-bin target or one explicit interpolation to the FFT grid
  -> optional design-grid compatibility policy
  -> minimum-phase FIR design
  -> no windowing under the registered default profile
  -> sample-peak or oversampled true-peak-estimate normalization
  -> explicit headroom verification
  -> response / latency / hash metadata
```

## Why This Exists

The point of this project is not just "generate a filter." It is to generate a filter with a clear output contract:

- same request -> same coefficients
- measurable headroom behavior
- verifiable realized response
- exported metadata that proves what was generated

That makes it suitable for portfolio work, repeatable lab workflows, and real convolution export pipelines.

It does not promise "better sound" on its own. Audible changes come from the target response, measurement quality, phase choice, headroom policy, and user-selected profile. The generator's job is to realize that request deterministically and prove what it produced.

## Output Contract

For a fixed version of the package, Python/NumPy/SciPy stack, input target, sample rate, FFT size, window, headroom, and profile, the tool guarantees:

- deterministic FIR coefficients
- deterministic coefficient SHA-256
- deterministic request fingerprint
- finite exported metadata
- explicit sample-peak and oversampled true-peak-estimate verification
- response-error validation against the requested target

The regression suite pins golden coefficient/request hashes for reference outputs and checks true-peak margin/error tolerances at `<= 1e-12` where numerical exactness is expected.

Coefficient hashes are generated from canonical little-endian float64 bytes. Bit-identical coefficients are guaranteed within the same Python/NumPy/SciPy/runtime stack; exported JSON records that runtime so cross-system differences can be audited instead of hidden.

Use `--reproducible` with `--export-json` when artifacts need byte-stable metadata across repeated runs. This sorts JSON keys, omits runtime timing, rejects non-finite JSON values, and rounds diagnostic floats to stable precision. Benchmark timing belongs in separate benchmark reports, not reproducible release metadata.

For reproducible release archives, use the repository build helper:

```bash
python scripts/build_reproducible.py --outdir dist
```

It builds the wheel/sdist with a fixed archive timestamp and normalizes ZIP/TAR metadata so repeated builds of the same tree produce identical artifact hashes. Set `SOURCE_DATE_EPOCH` or pass `--epoch` if your release process needs a specific timestamp.

## Project Layout

```text
src/fir_dsp/
  api.py                High-level pipeline entry points
  cli.py                Main command-line interface
  core.py               DSP primitives and verification helpers
  models.py             Typed result objects
  opra_cli.py           OPRA pack exporter
  analyze.py            FIR quality metrics and verdicts
  benchmark_analyze.py  Analyzer benchmark CLI
  txt_to_wav.py         Raw TXT impulse-response to WAV exporter
  validation.py         Centralized input validation

tests/
  test_api_cli_and_limits.py
  test_fir_correctness.py
  test_validation_and_determinism.py
```

Generated FIR packs, WAV files, response files, benchmark reports, build outputs, and local cache directories are intentionally ignored and should be regenerated locally.

## Installation

For development:

```bash
python -m pip install -e ".[dev]"
```

For direct source-tree CLI use without installation:

```bash
python -m fir_dsp.cli --help
```

## Input Formats

### One-column FFT-binned magnitude file

A one-column file is treated as a direct half-spectrum target with length:

```text
fft_size // 2 + 1
```

### Two-column response file

```text
frequency_hz  value
```

The value column can be interpreted as either:

- linear magnitude with `--input-scale linear`
- dB magnitude with `--input-scale db`

For two-column files, the response must extend to Nyquist for the requested sample rate.

## CLI Modes

The CLI exposes practical presets so you do not have to remember every knob.

### `--mode reference`

Good default for real exports.

- true-peak enabled
- oversample factor `8`
- no windowing
- design policy value `1`

### `--mode high_precision`

Use when you want the most conservative/highest-effort defaults.

- true-peak enabled
- oversample factor `8`
- no windowing
- design policy value `1`

### `--mode mart_reference`

Locked real-world preset for reference-chain exports.

- true-peak enabled
- oversample factor `8`
- no windowing
- design policy value `1`
- intended as the "stop tweaking and export the pack" mode

Manual arguments still override the preset where appropriate.

## CLI Examples

### Single-rate export

Create or provide your own two-column response file first; this example assumes
it is named `response.txt`.

```bash
python -m fir_dsp.cli \
  --response response.txt \
  --input-scale linear \
  --fft-size 1024 \
  --sample-rate 48000 \
  --headroom-db 6 \
  --mode reference \
  --out output/fir.txt \
  --export-json \
  --plot \
  --verbose
```

### Multi-rate export

This example also assumes a local `response.txt` input file.

```bash
python -m fir_dsp.cli \
  --response response.txt \
  --input-scale linear \
  --fft-size 1024 \
  --headroom-db 6 \
  --rates 44100 48000 96000 \
  --design-os 1 \
  --true-peak \
  --oversample-factor 8 \
  --out output/fir.txt \
  --export-json \
  --reproducible \
  --verbose
```

### Full standard PCM pack

```bash
python -m fir_dsp.cli \
  --input response.txt \
  --input-scale db \
  --fft-size 131072 \
  --headroom-db 9 \
  --all-pcm \
  --output fir.txt \
  --output-dir fir_pack \
  --mode reference \
  --export-json
```

### OPRA-backed export

```bash
python -m fir_dsp.cli \
  --opra-query "WH-1000XM5" \
  --opra-target oratory1990_harman_target \
  --opra-db "https://opra.roonlabs.net/database_v1.jsonl" \
  --fft-size 131072 \
  --headroom-db 9.60 \
  --true-peak \
  --profile default \
  --sample-rate 48000 \
  --out output/fir_48000.txt \
  --oversample-factor 8 \
  --design-os 1 \
  --window none \
  --export-json \
  --reproducible
```

### OPRA pack helper

```bash
python -m fir_dsp.opra_cli show "Audio Dest Ti"
python -m fir_dsp.opra_cli eq-pack http://opra.roonlabs.net/database_v1.jsonl WH-1000XM5 --target oratory1990_harman_target
python -m fir_dsp.opra_cli eq-pack http://opra.roonlabs.net/database_v1.jsonl WH-1000XM5 --target oratory1990_harman_target --target-sample-rate 48000
python -m fir_dsp.opra_cli tier3-pack http://opra.roonlabs.net/database_v1.jsonl WH-1000XM5 --target oratory1990_harman_target --source-rate 44100 --design-rate 48000 --chain-response roon_44100_to_48000_response.txt --resampler-name roon_measured
python -m fir_dsp.opra_cli profile-map "C:\path\to\database_v1_2.jsonl" --output opra_profile_directory_map.json --root fir_profiles --mkdirs
```

`opra_cli eq-pack` and `opra_cli eq-pack-all` are locked to the strict OPRA release path: `default` profile only, fixed FFT size `131072`, headroom `9.6` dB, true peak enabled, oversample factor `8`, design oversample `1`, default PCM rates, fixed artifact layout, and no windowing. `eq-pack` only accepts the OPRA database source, product query, required `--target`, and optional `--target-sample-rate`; release DSP settings cannot be overridden from this CLI.

`--target-sample-rate 48000` pins OPRA EQ evaluation to a fixed 48 kHz playback curve before the pack is discretized for each exported PCM rate. Use it for fixed-rate playback chains where upstream audio is resampled before the FIR is applied.

The OPRA pack manifest records the target projection separately from each FIR design rate: `measurement_domain`, requested `target_sample_rate`, resolved `reference_target_rate`, per-artifact `design_sample_rate`, and `projection_stage: pre_design`.

The OPRA pack gate is hard on per-rate target realization, profile, headroom, true peak, no-window, and artifact contracts. Cross-rate shape comparison is exported as `fir_cross_rate_consistency.json` from the canonical shared target.

`opra_cli tier3-pack` is a separate chain-specific path. It requires an explicit measured two-column resampler chain response file and writes outputs under `fir_profiles_tier3` by default. Tier 3 changes the design equation to `(resampler_chain * FIR) ~= requested_target`, records `projection_stage: post_chain_compensated`, includes the chain model and measured response hash in the request fingerprint, and should only be used for that exact playback chain.

Tier 3 chain-response files must start at `0 Hz`, be strictly increasing, and cover the active content band through `min(source_rate, design_rate) / 2`. Compensation is not attempted above that band; those bins use unity because there is no source content there. Measured magnitudes below `-120 dB` are rejected instead of clamped because inverse compensation through a null is numerically unstable. Chain closure error is reported only over the active source/content band. The manifest also records conditioning metrics such as max inverse gain and flags inverse boosts above `+20 dB`.

By default Tier 3 uses the exact measured inverse through the active band. If you explicitly want a narrow edge softening, pass `--chain-transition-hz 500`; this applies a fingerprinted cosine blend to unity below the active band edge. Tier 3 intentionally builds one design-rate FIR, not a multi-rate pack.

If you need to pin a specific source or use a local mirror, pass the database URL or path explicitly:

```bash
python -m fir_dsp.opra_cli show https://opra.roonlabs.net/database_v1.jsonl "Audio Dest Ti"
```

Each OPRA-generated export folder includes `selected_eq.txt`, `NOTICE_OPRA.txt`, and `ATTRIBUTION_OPRA.json` so attribution travels with the output pack.
The `profile-map` helper writes a reproducible JSON manifest for every valid EQ profile in the OPRA database and can optionally pre-create the corresponding directory tree. Paths are split by EQ id as `vendor/product/target`, and each manifest entry records whether that profile is the current default auto-selection or still requires explicit targeting.

### Supported aliases

- `--response` or `--input` for the FIR CLI
- `--out` or `--output` for the FIR CLI

### TXT to WAV

`fir-dsp-txt2wav` performs raw mono 32-bit float WAV export from a text impulse-response file. It does not apply hidden gain, embedded preamp processing, or int16 quantization.

```bash
python -m fir_dsp.txt_to_wav \
  --input output/fir_48000.txt \
  --output output/fir_48000.wav \
  --sample-rate 48000
```

Optional normalization is explicit:

```bash
python -m fir_dsp.txt_to_wav \
  --input output/fir_48000.txt \
  --output output/fir_48000.wav \
  --sample-rate 48000 \
  --normalize
```

### Analyzer

```bash
python -m fir_dsp.analyze \
  --fir output/fir_48000.txt \
  --sample-rate 48000 \
  --extended \
  --strict \
  --json-out output/fir_48000_analysis.json
```

### Target validation

When validating against a separate two-column target file, provide the target
value scale explicitly:

```bash
python -m fir_dsp.cli \
  --response response.txt \
  --input-scale linear \
  --fft-size 1024 \
  --headroom-db 6 \
  --out output/fir.txt \
  --validate-target \
  --target target_linear.txt \
  --target-scale linear
```

For dB-valued targets, use `--target-scale db`.

Benchmark the analyzer runtime:

```bash
python -m fir_dsp.benchmark_analyze \
  --fir output/fir_48000.txt \
  --sample-rate 48000 \
  --warmup 3 \
  --repeat 20 \
  --json-out output/fir_48000_benchmark.json
```

## Practical Defaults

For most real-world convolution exports, this is the best starting point:

```bash
--mode reference --headroom-db 6
```

Then only override settings when you have a reason to do so.

Notes:

- Use either `--window`, `--window-preset`, or `--no-window`, not more than one.
- `--window` accepts `none`, `hann`, `kaiser`, or `blackman`; the registered `default` profile only permits `none`.
- `--window none` is equivalent to `--no-window`.
- `--window-preset` accepts `safe`, `sharp`, or `minimal_ringing` for custom/non-default profile use; the registered `default` profile rejects these presets.
- `--design-os` is retained for compatibility and future extension; only `1` is currently supported. Values greater than `1` are rejected so metadata cannot imply a different design path than the one used.
- `--oversample-factor` affects true-peak measurement, not the design grid.
- `--export-json` writes latency, response error, and verification metadata.
- `--reproducible` requires `--export-json` and makes exported JSON stable by sorting keys, omitting runtime timing, and rounding diagnostic floats.
- `--audible-stress` requires `--export-json`; it adds optional stress-probe diagnostics to metadata.
- `--profile default` is the single registered profile. It requires no windowing, true-peak mode, trusted pipeline preamp handling, deterministic verification, and a 1.0 dB playback true-peak safety margin.
- OPRA `--target-sample-rate` pins EQ target evaluation to a fixed playback sample rate while preserving the same strict no-window default profile.
- OPRA source `gain_db` / preset preamp is recorded in exported `gain_traceability` metadata. When the EQ JSON is converted directly, that source preamp is baked into the target shape rather than applied again as a separate gain stage.

## Python API

```python
from fir_dsp.api import generate_fir_pipeline

result = generate_fir_pipeline(
    magnitude=[1.0] * 513,
    fft_size=1024,
    headroom_db=6.0,
    sample_rate=48000,
    true_peak=True,
    oversample_factor=8,
    design_oversample=1,
    return_details=True,
)

print(result.error.max_abs_error_db)
print(result.latency.peak_latency_ms)
print(result.verification.coeff_hash_sha256)
print(result.verification.fir_true_peak_dbfs)
```

## Exported JSON Metadata

The CLI metadata is meant to be auditable. It includes:

- sample rate
- FFT size
- input scale
- minimum-phase flag
- requested and normalization headroom
- window info
- latency summary
- response error summary
- coefficient hash
- target hash
- request fingerprint
- sample peak and true peak

Runtime fields such as generation time are diagnostics, not part of the signal contract. Use `--reproducible` for release artifacts where repeated runs should produce identical JSON.

## Tests

Install the development tools, then run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite covers deterministic output, headroom enforcement, interpolation and validation rules, CLI JSON export behavior, phase modes, and analyzer behavior.

## Licensing

The source code, command-line pipeline, Python API, tests, and documentation in this repository are licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Generated FIR filters, response exports, verification reports, benchmark reports, plots, and similar proof artifacts produced by this pipeline are licensed under CC BY 4.0 unless a generated artifact carries a more specific notice. See [ARTIFACTS_LICENSE.md](ARTIFACTS_LICENSE.md).

Generated artifacts are intentionally not committed to the source repository. Use the CLI examples above to regenerate FIR packs and verification outputs.

## OPRA Attribution

If you use the OPRA integration, review [NOTICE_OPRA.md](NOTICE_OPRA.md).

- OPRA repository code is MIT-licensed.
- OPRA dataset content is CC BY-SA 4.0.
- The OPRA authors also publish mirror-consumption guidance for reducing load on GitHub.

If an output is generated from OPRA data, preserve the generated OPRA notice and attribution files with the redistributed output.
