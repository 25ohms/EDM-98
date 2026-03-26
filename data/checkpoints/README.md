# Checkpoints

This directory documents the local checkpoint convention for optional EDMFormer inference.

Recommended local filename:

- `model.pt`
- `pretrained_msd.pt`
- `msd_stats.json`

Recommended behavior for the package:

1. use `--checkpoint` if provided
2. otherwise use `EDM98_CHECKPOINT` if set
3. otherwise look for `data/checkpoints/model.pt`

Do not commit large checkpoint binaries into git history unless you explicitly decide to distribute them that way.

Preferred publication options:

- Hugging Face model repository
- GitHub Release asset
- another documented external artifact host

When publishing the checkpoint, document:

- filename
- version
- checksum
- expected config compatibility

Additional inference assets expected by the current optional inference path:

- `msd_stats.json`
- `pretrained_msd.pt`

These correspond to MusicFM weights and should be documented alongside any setup instructions.

MuQ and MusicFM also rely on Hugging Face-backed upstream assets. By default, `EDM-98`
uses a local cache directory at:

- `.cache/huggingface/`

You can prefetch those assets once with:

```bash
python -m edm98.cli warm-cache
```

After the cache is populated, local-only reuse can be enforced with:

```bash
python -m edm98.cli warm-cache --offline
```

Large binary checkpoint assets in this directory should be tracked with Git LFS.
At present, this applies to:

- `model.pt`
- `pretrained_msd.pt`
