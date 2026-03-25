# Checkpoints

This directory documents the local checkpoint convention for optional EDMFormer inference.

Recommended local filename:

- `EDMFormer.safetensors`

Recommended behavior for the package:

1. use `--checkpoint` if provided
2. otherwise use `EDM98_CHECKPOINT` if set
3. otherwise look for `data/checkpoints/EDMFormer.safetensors`

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
