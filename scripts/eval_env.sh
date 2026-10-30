# Sourced by run_eval.sh. Checks the secrets the eval needs, without ever printing them.

# A pasted secret often carries a trailing newline. HTTP then rejects the Authorization header
# before any request is sent, and the SDK reports only "APIConnectionError: Connection error",
# which looks like a network failure. No key or model name can contain whitespace, so strip it.
clean_env() {
  local name value cleaned
  for name in "$@"; do
    value="${!name-}"
    cleaned="$(printf '%s' "$value" | tr -d '[:space:]')"
    if [ "$cleaned" != "$value" ]; then
      echo "warning: $name contained whitespace (often a trailing newline in the secret); removed it. Re-save the secret without it." >&2
      export "$name=$cleaned"
    fi
  done
}

# Fails early and clearly, instead of 30 cases failing one by one.
require_env() {
  local name
  for name in "$@"; do
    if [ -z "${!name-}" ]; then
      echo "error: $name is empty. In CI, pull requests from forks get no repository secrets." >&2
      return 1
    fi
  done
}
