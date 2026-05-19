# Inspect Examples

These commands assume a GitHub repository checkout.

If you are not using an installed `lynxes` command yet, replace `lynxes` with:

```bash
cargo run -p lynxes-cli --
```

## Inspect the smallest shared graph

```bash
lynxes inspect examples/data/example_simple.gf
```

## Inspect the weighted graph

```bash
lynxes inspect examples/data/example_weighted.gf
```

## Inspect the larger typed graph

```bash
lynxes inspect examples/data/example_complex.gf
```
