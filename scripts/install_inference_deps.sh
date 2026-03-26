#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
MUSICFM_DIR="${THIRD_PARTY_DIR}/musicfm"
HF_CACHE_DIR="${REPO_ROOT}/.cache/huggingface"

cd "${REPO_ROOT}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[inference]"

mkdir -p "${THIRD_PARTY_DIR}"
mkdir -p "${HF_CACHE_DIR}"
if [[ ! -d "${MUSICFM_DIR}/.git" ]]; then
  git clone https://github.com/minzwon/musicfm.git "${MUSICFM_DIR}"
fi

echo
echo "MusicFM source checked out at: ${MUSICFM_DIR}"
echo "Local Hugging Face cache directory: ${HF_CACHE_DIR}"
echo "Set MUSICFMPATH when running inference:"
echo "  export MUSICFMPATH=\"${MUSICFM_DIR}\""
echo
echo "Optional: prewarm MuQ and MusicFM cache assets once with:"
echo "  python -m edm98.cli warm-cache"
