# DEV-ONLY trace (not Zetaris, not a sample)

`DEV-ONLY-dev-q1-snow.*` is an early run against the DEV-ONLY DuckDB stand-in (`DATA_SOURCE=dev`, a small public Jan-Feb 2024 extract), made before the live Zetaris instance was up. It ran on older agent code and older checks, it is not replayable in SAMPLE_MODE, and none of its numbers are used anywhere in the submission. It is kept only as a record of development.

The three submission runs are `logs/recorded-*.jsonl`, with results in `output_examples/`.
