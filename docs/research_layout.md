# Research Layout

This repository now has a canonical, non-destructive browsing entrypoint:

- `research/`

The purpose is to separate:

- raw experiment outputs
- curated views
- project-level analysis packages
- focused study packages

without breaking old paths that existing scripts still use.

## Rebuild

```bash
python3 scripts/build_research_layout.py
```

## Design Rule

The `research/` tree is a generated workspace:

- `research/data/raw/`: raw runs grouped by collection, each with `canonical_runs/`,
  duplicate manifests, and a generated `README.md`
- `research/data/curated/`: curated symlink views and manifests
- `research/analysis/project_level/`: task atlas, reviews, follow-up analysis, and main summaries
- `research/analysis/studies/`: one folder per focused study, linking both spec and artifacts
- `research/metadata/`: machine-readable catalog and compatibility notes

Each raw-data collection gets an automatically generated condition summary based on discovered
`config.json` and `metrics*.csv` files where available. Exact duplicate runs are separated from
canonical runs using file-content hashes, and nonstandard or incomplete runs are surfaced in
`flagged_runs/` plus manifest files.
