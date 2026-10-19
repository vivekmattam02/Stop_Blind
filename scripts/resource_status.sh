#!/usr/bin/env bash
# One-glance resource picture for a GridNav run on the DGX Spark (GB10).
#
# GB10 uses UNIFIED memory: there is no separate VRAM pool, so nvidia-smi reports
# memory.total/used/free as [N/A]. GPU allocations and host allocations come out of
# the same 121.7 GB. Read "GPU memory" below as "the share of that pool the CUDA
# process currently holds", not as a separate card.
set -uo pipefail
C="${1:-geo_train_134k}"

printf '\n== GPU ==\n'
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader | awk -F', ' '{printf "  %s | util %s | %s | %s\n",$1,$2,$3"C",$4}'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
  | awk -F', ' '{printf "  cuda pid %s holds %s of the shared pool\n",$1,$2}'

printf '\n== container ==\n'
if docker ps --format '{{.Names}}' | grep -qx "$C"; then
  docker stats --no-stream --format "  {{.Name}}: cpu {{.CPUPerc}}  host-side mem {{.MemUsage}}" "$C"
else
  printf '  %s is not running\n' "$C"
fi

printf '\n== shared 121.7 GB pool ==\n'
free -g | awk 'NR==2{printf "  used %s GB | available %s GB | cache %s GB\n",$3,$7,$6}'

printf '\n== disk ==\n'
df -h $HOME/datasets/GridNav 2>/dev/null | awk 'NR==2{printf "  datasets: %s used of %s (%s)\n",$3,$2,$5}'
du -sh $HOME/datasets/GridNav/checkpoints/geometry_depth_cache_v1 2>/dev/null \
  | awk '{printf "  observation cache: %s\n",$1}'
printf '\n'
