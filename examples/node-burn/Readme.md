# node-burn: CPU/GPU GEMM tests

## Build container image

Following the main Readme, these commands have initially succeeded to build a container:

```
fcw container build --platform linux/arm64 -f env/Dockerfile.multistage --stage download -t node-burn:12.4.1-devel-ubuntu22.04-download .

fcw container push node-burn:12.4.1-devel-ubuntu22.04-download

fcw container build-remote node-burn:12.4.1-devel-ubuntu22.04-download \
    -f env/Dockerfile.multistage -t node-burn:12.4.1-devel-ubuntu22.04 \
    --stage build-offline \
    --enroot --wait
```

This should still be reviewed/double-checked.