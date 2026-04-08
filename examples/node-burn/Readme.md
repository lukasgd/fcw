# node-burn: CPU/GPU GEMM tests

## Build and deploy container image

All-in-one:

```bash
fcw container deploy node-burn --wait
```

Or as explicit steps:

```bash
fcw container build node-burn
fcw container push node-burn
fcw container build-remote node-burn --enroot --wait
```

## Run

```bash
fcw job submit node-burn
```
