# ProbeScout interface

After [preparing assets](../../../docs/assets.md), run here:

```sh
npm ci
npm run dev
```

The launcher starts the Web application and tuning API, using
`PROBESCOUT_PYTHON`, a project virtual environment, or `python`.

## Launch options

Run these commands from this Web directory, after installing dependencies and
the required [task ZIP and local images](../../../docs/manual_setup.md).
The task ZIP supplies saved rankings and Weight Tune inputs; matching HF features
and checkpoints additionally enable native probe updates.
Original images are required for galleries even when using HF embeddings.
For your own query/attributes, follow [New tasks](../../../docs/new_tasks.md)
to supply the JSON/CSV inputs, train and register a local task before launching.

- Combined development launcher: `npm run dev` starts the API on
  `127.0.0.1:8787`, waits for it, and starts the interface at
  `http://127.0.0.1:3000`. This is the default local entry point.
- Separate processes: run `npm run tuning:server` in one terminal and
  `npm run dev:web` in another. The frontend proxies `/api/tuning` to the API.
- Build commands: `npm run build` and `npm start` exist for the frontend.
  `npm start` does not launch the Python API. A complete production deployment
  also needs API routing and asset/runtime setup; that end-to-end deployment
  has not been verified in this release.

## Code and checks

- `scripts/tuning_models.py`: initial score and staged weight refinement.
- `scripts/tuning_supervision.py`: supervision, alignment and merging.
- `scripts/tuning_server.py`: task/session/job service and HTTP API.
- `scripts/unified_initial_baseline.py`: bank/evidence verification.
- `app/Dashboard.tsx`, `app/components/`, `app/lib/`: views and interactions.

```sh
npm run test:tuning
npm run test:unit
npm run build
```

These checks do not require a gallery. With the prepared Web assets installed,
run `npm run test:assets` for Main17 binary, identity and query alignment checks.
The retained core tests cover fusion/feedback numerics, frozen Val and gates,
probe snapshots, the update/refinement workflow, and Development/Frozen Test
metric separation. See [validation](../../../docs/validation.md) for results.

The paper uses `weight_staged` with frozen probes/gates. New feedback stays in
runtime. Historical sessions and derived models are excluded from publication.
