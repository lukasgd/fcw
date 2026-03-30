ARG BASE_IMAGE=ubuntu:24.04
ARG DOWNLOAD_IMAGE=download

FROM ${BASE_IMAGE} AS download

WORKDIR /workspace/fcw-basic

COPY . /workspace/fcw-basic


FROM ${DOWNLOAD_IMAGE} AS download-copy

FROM ${BASE_IMAGE} AS build-offline

COPY --from=download-copy /workspace/fcw-basic /workspace/fcw-basic

WORKDIR /workspace/fcw-basic

RUN cat data/raw/test.txt