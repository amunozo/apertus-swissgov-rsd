# Hack Apertus — Track 2A (UZH: SwissGov-RSD)

> **Winner of Hack Apertus 2026, Track 2A** ([announcement](https://lnkd.in/p/ebDrTf2X)).

> **Submission root: [`track_2a/`](track_2a/)** — see [`track_2a/README.md`](track_2a/README.md) and [`track_2a/technical_report.md`](track_2a/technical_report.md). To run the submission in Docker from a clean checkout: `cd track_2a && make run`.

Repository for the [Hack Apertus](https://hackapertus.ch/) Track 2A submission addressing the **University of Zurich (UZH)** challenge (*Detecting Cross-Lingual Semantic Differences in Swiss Government Websites*).

## Structure (`track_2a/`)

| Path | What it is |
| --- | --- |
| `track_2a/README.md` | Overview, headline results, and instructions to run |
| `track_2a/technical_report.md` | Full technical report: architecture, evaluation, error analysis, limitations |
| `track_2a/Makefile` | `make run` builds and runs the CPU Docker pipeline |
| `track_2a/src/` | Core Python packages (`rsd` and `official_eval`) |
| `track_2a/data/` | Precomputed Apertus feature archives and frozen test predictions (41.8 MB; max. 100 MB) |
| `track_2a/docs/` | Experiment write-ups and interactive token-level HTML heatmaps |

## Run it

From `track_2a/` on a clean checkout:

```bash
cd track_2a && make run
```

`make run` runs the project in a CPU-only Docker container without requiring a GPU, model weights, or API keys.

## License

All Hack Apertus projects are open-sourced. Code is licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)); documentation and [`track_2a/technical_report.md`](track_2a/technical_report.md) are licensed under CC BY 4.0. Please also see the Hack Apertus Terms & Conditions: https://hackapertus.ch/terms-and-conditions
