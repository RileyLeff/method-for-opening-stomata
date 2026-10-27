# Original data

Start with the [data guide and field dictionary](../docs/DATA.md) and
[run index](run_index.csv). Original names and bytes are preserved, including
typos and excluded runs. All raw files are ordinary Git objects, not LFS pointers.

`manifest.csv` records each original file's SHA-256 and byte size. The index
extracts instrument versions from log headers, maps cohort sample IDs, and carries
QC status. Program association is inferred, not verified by an instrument-source
hash. Filename-derived species labels are explicitly named as such.

The original flash ZIPs and extracted event files intentionally duplicate content
for archival and inspection convenience. The file count is not an observation
count. Do not edit raw files to implement exclusions.

```sh
python3 scripts/release_manifest.py --check --raw-only
```

Run that command from the repository root. Metadata tables and README files are
version-controlled separately; the raw manifest excludes its own index/documentation.
CC BY 4.0 applies; see [license](../LICENSE-DATA.md).
