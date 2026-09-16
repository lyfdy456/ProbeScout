# Release validation

## Core tests

The release retains 24 test files and two support files, reduced from 104 test
and support files on 2026-09-16. These are executable correctness checks, not
experiment results. Tests generate synthetic data and temporary models; running
the application does not require running the tests.

| Suite | Test files | Latest result | Coverage |
|---|---:|---|---|
| Probe learning | 7 | 38 passed | Eight probe implementations, checkpoint loading, fixed holdouts, gate calibration, supervision merging, Main17/ablation entries and credential exclusions |
| Web backend | 9 | 79 passed | Fusion and feedback numerics, frozen gates, immutable Val, probe snapshots, session isolation and the update-to-refinement workflow |
| Frontend | 7 | 36 passed | Score/weight alignment, scoped metrics and Development/Frozen Test separation |
| Optional Main17 assets | 1 | Passed for all 17 local task bundles | Binary checksums, image IDs, attribute order and fixed queries |

Run the synthetic suites after installing the documented dependencies:

```sh
# Repository root
cd probe_learning
uv run python -B -m unittest discover -s tests
cd ../visual_analytics/pcp_analyze/web
npm run test:tuning
npm run test:unit
```

The optional `npm run test:assets` requires prepared Web assets. Set
`PROBESCOUT_ASSET_WEB` to another Web directory to validate existing assets
without copying them into the repository.

`development-metrics.test.mjs` renders the result components and checks that
Development contains no Test metrics, including closed details, historical
Test-only runs and unknown evaluation scopes. Both refinement schedules and
both probe sources are covered. Test metrics remain in the read-only Frozen
Test panel.

Layout, typography, temporary preview fixtures and historical exploration tests
were removed from this release copy after a verified local backup. Some retained
files were narrowed to their core numerical and isolation cases. Application
and training code were unchanged by this cleanup.

## Earlier validation

Before the test reduction, all 320 frontend tests and four credential-boundary
tests passed on 2026-09-16, together with a production Web build. These are
historical full-suite counts, not the size of the retained suite above.

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

## Scope of these checks

Python and Node checks used existing compatible local dependency installations.
The retained suites passed again after the reduction (38 learning, 79 backend,
36 frontend, and one Main17 asset integration check). No tests were skipped.
The build was not repeated for this test/documentation-only cleanup. Earlier
temporary dependency links and build output were removed after verification.

Full paper retraining, complete image extraction, an actual clean dependency
download/install, and an interactive gallery/feedback session were not performed.
Source tests and a successful build do not make a fresh clone an asset-complete
demo; see [asset setup and distribution](assets.md) for the separate packages.

The pre-cleanup publication snapshot contains 585 authored files. A separate
original-source snapshot contains 597 files; their original bytes remain
unchanged. Two hand-written development-server boundary files omitted by the
earlier source copier were recovered from the original and backed up separately.
