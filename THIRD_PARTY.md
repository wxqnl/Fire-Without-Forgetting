# Third-party sources and attribution

This repository contains adapted snapshots and upstream source copies. Original copyright notices and license files remain in their respective directories. See [docs/sources.json](docs/sources.json) for recorded source URLs, branch names, source dates, and retrieval times.

| Upstream project | Source | License file |
|---|---|---|
| Spiking-WM | https://github.com/Brain-Cog-Lab/Spiking-WM | [Apache-2.0](sources/Spiking-WM/LICENSE) |
| Recall2Imagine | https://github.com/chandar-lab/Recall2Imagine | [MIT](sources/Recall2Imagine/LICENSE) |
| DreamerV3 | https://github.com/danijar/dreamerv3 | [MIT](sources/dreamerv3/LICENSE) |
| BSuite | https://github.com/google-deepmind/bsuite | [Apache-2.0](sources/bsuite/LICENSE) |
| POPGym | https://github.com/proroklab/popgym | [MIT](sources/popgym/LICENSE.md) |
| Memory Maze | https://github.com/jurgisp/memory-maze | [MIT](sources/memory-maze/LICENSE) |

MCN and persistent-SSM snapshots build on Spiking-WM; R2I snapshots build on Recall2Imagine; Dreamer snapshots build on DreamerV3. Local environment adapters, evaluation/counter/save changes, MCN interventions and the persistent-state model are described in [the code map](docs/CODE_MAP.md) and the audit/patch files in docs.

Upstream figures, benchmark scores and README material inside source directories are upstream reference material, not results newly produced by this project. Project-generated experiment records are under results/ and the associated evidence files under docs/.

Some upstream repositories contain further third-party notices. Those files are retained with the source. This index does not replace the original license terms or relicense third-party files.
