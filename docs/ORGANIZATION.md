# Release organization decision

October 27, 2026. Riley requested a complete recoverable snapshot, consultation
with Claude Opus 5.5, and a GitHub repository containing the data as well as code.

## Decision

Preserve data/, outputs/, src/, bp/, and the flat scripts/ layout. Add explicit
navigation, manifests, QC documentation, tests, and licenses rather than refactor
the scientific code before the deadline. Raw data, including exclusions, stay
in ordinary Git. Scientific outputs stay in Git except the oversized generated
cohort points CSV, which uses Git LFS. Archive ZIPs that duplicate included files,
poster branding, commercial fonts, vendor manuals, and temporary render products
stay in the private snapshot.

The original workspace snapshot is commit `715bac5`, locally tagged
`snapshot/pre-release-2026-10-27`. It is NOT for public push. The release branch
`codex/open-science-release` starts at code-only commit `0f788d4`, in a separate
worktree. No history was reset or rewritten. A pre-push guard rejects the private
snapshot ancestry (including its tag) and oversized non-LFS blobs, then runs LFS.

The remote is still private. This setup does not itself publish anything or
complete the protocols.io/Zenodo deliverables. Publication requires an explicit
push, a remote clone/LFS verification, and a final release decision.

## Consultation

Claude Opus 5.5 was invoked through Claude Code 2.1.305 with model
`claude-opus-5-5`, session `689caae5-d4a9-4e87-bfc3-0b88abe6cb79`.
It inspected the original repository with a read-only task instruction and normal
inspection tools; it made no edits. Raw CLI logs remain outside the repository.

We agreed on preserving paths, shipping raw data, LFS for one table, a separate
release worktree based on pre-snapshot history, and clear historical/current
program labels. We retained the existing local snapshot tag but block its push,
rather than deleting it. We also split heavy Python dependencies into optional
extras to keep basic data inspection and CI lightweight.

## Confirmed by owner

Riley approved MIT code licensing and CC BY 4.0 data/documentation licensing and
confirmed collaborator release clearance on October 27. Instrument versions were
recovered from all 15 cohort headers, rather than assuming latest firmware.

## Remaining release work

- Confirm any additional sponsor/site archival requirements (for example an EDI
  deposit) separately; the repository does not establish those obligations.
- Complete and publish the operating protocol and technical report/Zenodo record.
- Obtain the DOI, then add it to citation and cross-links without inventing one.
- Enable/check LFS archive inclusion and verify the actual remote download.
- Publish public URLs and send them to Astera before the October 30 deadline.

No new biological measurements or Bayesian sampling are required to organize this
release. Validation status is recorded in VALIDATION.md after local checks.
