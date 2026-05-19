# Convert Examples

These commands assume a GitHub repository checkout.

If you are not using an installed `lynxes` command yet, replace `lynxes` with:

```bash
cargo run -p lynxes-cli --
```

## Convert `.gf` to `.gfb`

```bash
lynxes convert examples/data/example_simple.gf example_simple.gfb --compression zstd
```

## Inspect the converted file

```bash
lynxes inspect example_simple.gfb
```

## Convert `.gfb` back to `.gf`

```bash
lynxes convert example_simple.gfb example_simple_roundtrip.gf
```
