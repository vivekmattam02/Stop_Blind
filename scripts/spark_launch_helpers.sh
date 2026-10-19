#!/usr/bin/env bash

# Confirm that a backgrounded `docker run` has registered its named container before the launcher
# returns. Every caller must first verify that its image exists locally, so this handshake never
# waits for an image pull. A stage that finishes unusually quickly is accepted only when its
# verification artifact already exists.
wait_for_container_registration() {
  local container_name="$1"
  local docker_client_pid="$2"
  local verification_path="$3"
  local log_path="$4"
  local timeout_seconds="${GRIDNAV_CONTAINER_REGISTRATION_TIMEOUT_SECONDS:-120}"
  local waited_ticks=0
  local maximum_ticks=$(( timeout_seconds * 10 ))

  while (( waited_ticks < maximum_ticks )); do
    if docker inspect "${container_name}" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -f "${verification_path}" ]]; then
      return 0
    fi
    if ! kill -0 "${docker_client_pid}" 2>/dev/null; then
      local exit_status=0
      if wait "${docker_client_pid}"; then
        exit_status=0
      else
        exit_status=$?
      fi
      echo "Background docker client exited before ${container_name} registered (status ${exit_status})." >&2
      if [[ -f "${log_path}" ]]; then
        tail -n 40 "${log_path}" >&2
      fi
      return 1
    fi
    sleep 0.1
    waited_ticks=$(( waited_ticks + 1 ))
  done

  echo "Container ${container_name} did not register within ${timeout_seconds}s." >&2
  echo "The required image was already local, so the launcher is stopping before reporting success." >&2
  return 1
}
