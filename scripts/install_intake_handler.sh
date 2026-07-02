#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/research/skills/crelat-intake-handler"
skills_root="${CODEX_HOME:-$HOME/.codex}/skills"
target="$skills_root/crelat-intake-handler"

mkdir -p "$skills_root"
if [[ -L "$target" ]]; then
  current="$(readlink -f "$target")"
  if [[ "$current" == "$source_dir" ]]; then
    echo "CRELAT intake handler already installed: $target"
    exit 0
  fi
  echo "Refusing to replace symlink to unrelated source: $target -> $current" >&2
  exit 1
fi
if [[ -e "$target" ]]; then
  echo "Refusing to replace existing path: $target" >&2
  exit 1
fi
ln -s "$source_dir" "$target"
echo "Installed CRELAT intake handler: $target -> $source_dir"
