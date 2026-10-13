# Bundled EMP dependency provenance

DyFence's release build uses the source snapshots in this directory instead
of downloading EMP at build time. The snapshots contain only the public source,
CMake metadata, README, and license files required to build DyFence. Nested Git
repositories, upstream tests, generated binaries, and build caches are omitted.

The tree SHA-256 values below are computed from a sorted list of every bundled
file and its SHA-256 digest:

```bash
cd third_party/<component>
find . -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 \
  | shasum -a 256
```

## emp-tool

- Upstream: https://github.com/emp-toolkit/emp-tool
- Base commit: `802b5d4fb7cc7fcaadd411cd6aa5e72ed4dd57fd`
- Local changes: none
- License: `third_party/emp-tool/LICENSE`
- Tree SHA-256: `e0daf3345aea5d9aef76841c7c7cd449f8f07f1550d0b53ab1a3735c4c43a5e0`

## emp-ot

- Upstream: https://github.com/emp-toolkit/emp-ot
- Base commit: `a603ca0c77fcda37b9d088bd692111f67a4bef96`
- Local changes: none
- License: `third_party/emp-ot/LICENSE`
- Tree SHA-256: `823981c890a55e035c127924f0277f5fe3fda2affaa2f4927164750e4cedfb5c`

## emp-zk

- Upstream: https://github.com/emp-toolkit/emp-zk
- Base commit: `73ab193a923d4c122be5a4f6bc1fe4f617966b02`
- Local changes: DyFence-specific emp-zk-math extensions and the CMake source
  list required to build those extensions
- License: `third_party/emp-zk/LICENSE`
- Tree SHA-256: `3c2771695823886d68cf94eb8f8e4cc632e22d5f28df6d19d75f455ab6079277`
