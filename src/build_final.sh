#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

output=clusters_balanced_l2_96k_final
resume=()
heads=()
if [[ -f "$output/build.json" ]]; then
    resume=(--resume)
    # Keep the original parameters when resuming; cached heads no longer need their import source.
    imported=$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["parameters"].get("heads_from", ""))' "$output/build.json")
    if [[ -n "$imported" ]]; then
        heads=(--heads-from "$imported")
    fi
fi
client=client
if [[ ! -d "$client" && -d ../client ]]; then client=../client; fi

.venv/bin/python -u clustering.py vectors \
    --output "$output" --clusters 96000 --sample-size 2500000 \
    --batch-size 65536 --threads 4 --brotli-centroids \
    "${heads[@]}" "${resume[@]}"

.venv/bin/python -u -m balanced_l2.export_posting_quantization --data "$output" --workers 4
.venv/bin/python -u prepare_client_release.py --data "$output" --meta meta --client "$client"

printf '%s\n' 'Final dataset ready locally. Upload and verify every data batch before publishing the client.'
