# Examples

The `examples/` tree is where LynxesMojo stays closest to runnable code. This layer is not trying to replace the original Lynxes documentation, and it is not trying to act like a reference manual. The job of these files is simpler than that: each example should answer one small practical question with code that can actually be run.

That is why the examples are split by role instead of being treated as one flat folder. Some files are meant to be read in order by someone who is still building a mental model of the library. Others are closer to recipes: you already know what Lynxes is, and you want to see the smallest complete example for one task. The test suite also leans on these files, so the structure needs to be stable enough for smoke tests to point at the same source of truth.

## Layout

- `data/` holds the shared graph files used across examples and tests.
- `python/tutorials/` holds the small numbered walkthrough examples that make sense to read in sequence.
- `python/recipes/` holds task-oriented Python examples that stand on their own.
- `cli/` holds command-oriented examples for the terminal surface.
- `rust/tutorials/` holds the Rust-side walkthrough examples that mirror the broad progression of the Python tutorial lane.

## Shared Data

The graphs under `data/` are intentionally small and named for the role they play:

- `example_simple.gf` is the default first graph. It is the one most tutorial examples should prefer.
- `example_weighted.gf` exists for weighted shortest-path and route-style examples.
- `example_complex.gf` is for richer inspection, labels, edge types, and algorithm outputs that need a slightly less toy graph.

This folder should stay disciplined. If a new graph exists only to trigger a parser edge case, it usually belongs in tests or fixtures instead. Files that live here should earn their place by being reused across languages, examples, or smoke tests.

## Python Examples

The Python examples are split into two lanes.

The tutorial lane is short on purpose. Those files are the ones that make sense to read in order:

- `python/tutorials/01_read_and_inspect.py`
- `python/tutorials/02_lazy_expand.py`
- `python/tutorials/03_first_algorithm.py`

The recipe lane is different. Those files are not trying to teach the whole library in sequence. Each one is a compact, task-shaped example:

- `python/recipes/pagerank.py`
- `python/recipes/community_detection.py`
- `python/recipes/io_roundtrip.py`

That split matters because numbered files imply a learning order, while task-named files imply independent reuse. Keeping both in one flat directory makes the role of each file less clear than it needs to be.

## CLI Examples

The CLI examples stay command-oriented:

- `cli/inspect.md`
- `cli/query.md`
- `cli/convert.md`

These are still examples, not reference pages. They should show a realistic input, the command to run, and the thing you should check afterward.

## Rust Examples

The Rust examples follow the same broad progression as the Python tutorial lane, but they stay idiomatic to the Rust surface:

- `rust/tutorials/01_read_gf.rs`
- `rust/tutorials/02_lazy_collect.rs`
- `rust/tutorials/03_first_algorithm.rs`

The goal is not perfect line-by-line symmetry. The goal is that someone comparing surfaces can see the same graph and the same kind of task represented across both languages.

## Writing Rules

Examples in this tree should stay small, runnable, and single-purpose. A good example shows one complete task, includes the imports and setup it needs, and prints enough output that a user can tell whether it worked. It can explain one or two important transitions, such as why `.collect()` matters, but it should stop well before turning into a guide.

That boundary is important. The upstream Lynxes docs teach the full library. These examples are the executable layer for this toy checkout.
