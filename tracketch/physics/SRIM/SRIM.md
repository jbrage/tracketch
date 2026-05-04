# SRIM Data

Raw SRIM `.dat` files live in:

- `tracketch/physics/SRIM/_raw/bragg_0/`
- `tracketch/physics/SRIM/_raw/bragg_3/`
- `tracketch/physics/SRIM/_raw/bragg_5/`

Processed CSV tables are generated into:

- `tracketch/physics/SRIM/_processed/bragg_0/`
- `tracketch/physics/SRIM/_processed/bragg_3/`
- `tracketch/physics/SRIM/_processed/bragg_5/`

## Add data

Add raw SRIM output files to the matching `bragg_<pct>` folder.

Expected file names:

- `CR39_1H.dat`
- `water_1H.dat`
- `CR39_4He.dat`
- `water_4He.dat`
- etc.

## Regenerate processed tables

From the repository root:

```bash
python -c "from tracketch.physics.SRIM.SRIM_lib import generate_SRIM_dfs; [generate_SRIM_dfs(pct) for pct in (0, 3, 5)]"
```

To regenerate only one variant:

```bash
python -c "from tracketch.physics.SRIM.SRIM_lib import generate_SRIM_dfs; generate_SRIM_dfs(3)"
```
