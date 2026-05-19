# Query Examples

These commands assume a GitHub repository checkout.

If you are not using an installed `lynxes` command yet, replace `lynxes` with:

```bash
cargo run -p lynxes-cli --
```

## Summary view with no seed

```bash
lynxes query examples/data/example_simple.gf
```

## Two-hop traversal from a seed node

```bash
lynxes query examples/data/example_simple.gf --from alice --hops 2 --direction out
```

## Restrict traversal to one edge type

```bash
lynxes query examples/data/example_simple.gf --from alice --hops 2 --edge-type KNOWS --direction out
```

## Use a richer output view

```bash
lynxes query examples/data/example_complex.gf --from c1 --hops 1 --direction out --view info
```
