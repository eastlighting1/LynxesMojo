# Python Examples

The Python examples are split into two lanes because they serve two slightly different jobs.

The tutorial lane is meant to be read in order by someone who is still getting oriented. Those files are small, incremental, and deliberately numbered:

- `tutorials/01_read_and_inspect.py`
- `tutorials/02_lazy_expand.py`
- `tutorials/03_first_algorithm.py`

The recipe lane is for task-oriented examples that stand on their own:

- `recipes/pagerank.py`
- `recipes/community_detection.py`
- `recipes/io_roundtrip.py`

All of these examples assume a repository checkout so they can read the shared graph files under `examples/data`. That is a deliberate tradeoff. It keeps the docs, examples, and smoke tests anchored to the same small set of canonical example graphs instead of letting each surface drift into its own private sample data.
