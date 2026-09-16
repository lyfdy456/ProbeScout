# ProbeScout interface

After [preparing assets](../../../docs/assets.md), run here:

```sh
npm ci
npm run dev
```

The launcher starts the Web application and tuning API, using
`PROBESCOUT_PYTHON`, a project virtual environment, or `python`.

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
