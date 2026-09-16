# Release validation

Follow-up on 2026-09-16: Development result rendering now excludes all Test
metrics, including closed details and historical Test-only runs. Test metrics
are displayed in the read-only Frozen Test panel. All 320 frontend tests and
four credential-boundary tests passed, and the production Web build passed.
Rendered component tests cover both refinement schedules, both probe sources,
missing/unknown evaluation scopes, legacy runs, and unavailable Test metrics.

The five VQA YAML credential fields and `.env.example` are empty. No private
environment files or original VQA credential strings were found in the release.
The release checker now also rejects nonempty credential fields, private `.env`
files, and recognizable tokens in documentation. Detection is a local check,
not an upload operation or a guarantee against every possible secret format.

The following records describe the preceding paper-cleanup validation.

Validated on 2026-09-15 against the publication copy.

| Check | Result |
|---|---|
| Probe learning: synthetic training, cache contracts, paper entries and gate numerics | 124 tests passed |
| Web backend: fusion, feedback, isolation, frozen evidence, jobs and bundle scope | 279 tests passed |
| Frontend and development-server contracts | 316 tests passed |
| Main17 asset integration | Passed for all 17 existing task bundles |
| Production Web build | `npm run build` passed |
| Fresh-directory npm lock validation | Offline `npm ci --dry-run --ignore-scripts` passed |
| Python dependency lock | `uv lock --check --offline` passed |
| Source boundary | Python/JSON syntax, package/lock consistency, Main17 scope and publication exclusions passed |

## Numerical reproduction

The public commands read the existing frozen evidence without copying features,
checkpoints or gallery arrays. Every reported Test/Gallery AP and F1, selected
Validation cutoff, and score-vector SHA-256 matched the original results exactly:

- Table 2: 204 rows, covering 17 tasks and 12 methods.
- Table 5: 68 rows, covering 17 tasks and four component variants.

Full-model macro Test AP is `0.8320685393249247`; Gallery AP is
`0.8232782494346567`. Both commands save Validation choices before evaluation.

The asset check validates binary file sizes/checksums, unique image IDs,
attribute order and fixed query-image alignment. It can use assets in another
Web directory through `PROBESCOUT_ASSET_WEB`, without copying them into source.

## Scope of these checks

Python and Node tests/builds used existing compatible local dependency
installations. The temporary Node dependency link and generated build output
were removed after verification. The npm lockfile was repaired to include D3
and its types and to remove unused hosting/database dependencies.

Full paper retraining, complete image extraction, an actual clean dependency
download/install, and an interactive gallery/feedback session were not performed.
The asset repositories have not been published. Source tests and a successful
build do not make a fresh clone an asset-complete demo.

The pre-cleanup publication snapshot contains 585 authored files. A separate
original-source snapshot contains 597 files; their original bytes remain
unchanged. Two hand-written development-server boundary files omitted by the
earlier source copier were recovered from the original and backed up separately.

Historical full-catalogue/hosted-site checks were replaced by explicit Main17
asset checks. Stale UI source assertions were aligned with the existing UI; the
application's feedback controls and scoring formulas were not changed to make
those assertions pass. Existing numerical, data-isolation and provenance tests
remain part of the suite.
