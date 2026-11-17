# ADR-001 — v1 scope and bioenergetic model

**Status:** accepted · 2026-11-16

## Context
The project began as "IYD — complex society simulator" on K2-18 b: a 38-line script whose inheritance step never ran (`return False`). The companion "Zoo" genomics project was archived because most of its target species (e.g. *Andrewsarchus*, *Daeodon*, *Dinocrocuta*) cannot have surviving DNA.

## Decisions
1. **Functional traits, not DNA.** Agents carry mass, insulation and locomotion. Every advantage costs energy, and every scaling exponent cites a source.
2. **v1 scope: terrestrial endothermic vertebrates on Earth.** There is a fossil pattern to compare against (Smith et al. 2010).
3. **K2-18 b moves to v2.** It needs an `endothermy` trait and an aquatic medium; placing terrestrial mammals in a hypothetical ocean would assume the answer.
4. **Structure of Arrays in NumPy** for speed.
5. **Fitness emerges from the population.** It is never computed from offspring counts of lineages that later go extinct.
6. **Offspring endowment scales with the offspring's own mass.** Copying the parent's energy secretly favoured downward mass mutations; this was found and fixed during v1.
7. **Well-mixed food pool for the experiments.** A fixed-cell grid caps intake for large animals unless home range scales with mass, which is deferred to v2.
8. **Fur effectiveness is limited by body size** (Scholander et al. 1950). Without this limit, small animals reached perfect insulation for free.
9. **Winter means cold AND scarce**, through a joint food and temperature cycle.
10. **"Species" means a phenotypic cluster (DBSCAN)**, and is described as such.
11. The repository is renamed **bio-evolution-sim**.

## Consequences
- The hypothesis is testable and the code stays simple.
- Cold and seasonality robustly increase body size: 81/81 parameter combinations.
- Warmth does not shrink bodies, because heat stress scales like metabolism. This is documented as a limitation, and a heat-dissipation limit is the top v2 item.
- Magnitudes depend on calibration; only the directions are claimed.

## Rejected alternatives
| Alternative | Reason |
|---|---|
| Simulating DNA letters | Not connected to survival in an ecological model |
| All taxa at once | Incompatible spatial scales; impossible to validate |
| `if climate:` rules | Circular validation |
| 2D grid for the v1 results | Penalises large bodies (see decision 7) |
