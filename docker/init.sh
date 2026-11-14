#!/command/with-contenv sh
# shellcheck shell=sh
# ColonyMind's setup at every container start, installed as /opt/colonymind/docker/init.
#
# An s6-rc oneshot (colonymind-init): it runs after s6's base bundle, which holds Hermes' own boot step (the hermes
# user remapped to HERMES_UID/HERMES_GID, Hermes' home owned by it and seeded), and the sidecar service depends on
# it, so a failure here leaves the sidecar down instead of serving an unconfigured instance. The reason is in
# `docker logs`.
#
# 1. The instance directory ($COLONYMIND_HOME, a volume of its own) belongs to the hermes user.
# 2. With COLONYMIND_CONTAINER_MODEL_URL, Hermes' model is that OpenAI-compatible endpoint serving
#    COLONYMIND_CONTAINER_MODEL over chat completions, set with Hermes' own `hermes config set`. The key, when there
#    is one, stays in COLONYMIND_CONTAINER_MODEL_KEY: Hermes' config names the variable, never the value. A key
#    Hermes holds belongs to the endpoint it was given for: moved to another host with no key given, Hermes' key is
#    removed, never sent to the new one. Init is given the same endpoint, so there is one, and a changed one is
#    followed by both: the router's own file is pointed there in place (docker/endpoint.py). The router's key is
#    the one given for that endpoint, or none: never a key Hermes' .env holds. Without it, a new instance is only
#    set up once Hermes has a model of its own: the config Hermes seeds names a hosted placeholder no one chose.
# 3. An existing instance is brought up to this image's version (`colonymind upgrade`), then `colonymind init`
#    applies the answers in the environment (COLONYMIND_INIT_*). Both are idempotent: a restart with the same
#    environment changes nothing.
set -eu

# The config Hermes copies into its home on first start (its stage2 hook), untouched until someone sets a model.
seed=/opt/hermes/cli-config.yaml.example
# The sidecar's interpreter, and the helper it runs to work out what the container's model endpoint replaces.
python=/opt/colonymind/.venv/bin/python
endpoint=/opt/colonymind/docker/endpoint.py

say() { echo "[colonymind] $*"; }

# Everything ColonyMind writes belongs to the hermes user, as Hermes' own data does.
as_hermes() {
    if [ "$(id -u)" = 0 ]; then
        s6-setuidgid hermes "$@"
    else
        "$@"
    fi
}

home="${COLONYMIND_HOME:?COLONYMIND_HOME names the instance directory}"
hermes_home="${HERMES_HOME:-/opt/data}"
# The hermes user's home, as in Hermes' own services (with-contenv hands over /root).
export HOME=/opt/data

if [ -L "$home" ]; then
    say "refusing to set up $home: it is a symbolic link; mount a directory there"
    exit 1
fi

model_url="${COLONYMIND_CONTAINER_MODEL_URL:-}"
model="${COLONYMIND_CONTAINER_MODEL:-}"
if [ -n "$model_url" ] && [ -z "$model" ]; then
    say "COLONYMIND_CONTAINER_MODEL_URL is set without COLONYMIND_CONTAINER_MODEL: name the model that endpoint serves"
    exit 1
fi
if [ -z "$model_url" ] && [ -z "${COLONYMIND_INIT_MODEL_URL:-}" ] && [ ! -f "$home/colonymind.yaml" ] \
        && cmp -s "$hermes_home/config.yaml" "$seed"; then
    say "Hermes has no model yet (its config.yaml is still the example it seeds): set COLONYMIND_CONTAINER_MODEL_URL" \
        "and COLONYMIND_CONTAINER_MODEL to one OpenAI-compatible endpoint, or configure Hermes' model (hermes model)" \
        "and restart; the sidecar starts once it is set up"
    exit 1
fi

mkdir -p "$home"
if [ "$(id -u)" = 0 ]; then
    uid="$(id -u hermes)"
    gid="$(id -g hermes)"
    # After a remap (HERMES_UID) the files still carry the old owner: hand the whole directory over, never
    # following a link out of it.
    if [ -n "$(find "$home" \( ! -uid "$uid" -o ! -gid "$gid" \) -print -quit)" ]; then
        say "giving $home to the hermes user ($uid:$gid)"
        chown -hR "$uid:$gid" "$home"
    fi
    chmod 0700 "$home"
fi
cd "$home"

if [ -n "$model_url" ]; then
    # Hermes reads its model's key from model.api_key, else from the legacy model.api. Each is checked before the base
    # URL moves (the endpoint a key was given for is the one the config names now): model.api_key when the variables
    # give no key (one they give replaces it below), model.api always (a key given only shadows it, and emptying the
    # variable later would bring it back).
    fields="api"
    [ -n "${COLONYMIND_CONTAINER_MODEL_KEY:-}" ] || fields="api_key api"
    for field in $fields; do
        verdict="$(as_hermes "$python" "$endpoint" hermes-key "$hermes_home" "$model_url" "$field")" || {
            say "cannot tell which endpoint Hermes' model.$field belongs to (above); the sidecar starts once it can"
            exit 1
        }
        case "$verdict" in
            keep) ;;
            moved)
                as_hermes hermes config unset "model.$field"
                say "Hermes' model.$field was for its earlier endpoint and is removed, not sent to $model_url;" \
                    "set COLONYMIND_CONTAINER_MODEL_KEY if that endpoint needs a key";;
            removed)
                as_hermes hermes config unset "model.$field"
                say "COLONYMIND_CONTAINER_MODEL_KEY is empty now: Hermes' model.$field, which named it, is removed," \
                    "and the router's file follows";;
            *)
                say "unexpected answer from $endpoint: $verdict"
                exit 1;;
        esac
    done
    as_hermes hermes config set model.provider custom
    as_hermes hermes config set model.base_url "$model_url"
    as_hermes hermes config set model.default "$model"
    # The endpoint speaks the OpenAI-compatible chat completions route (the router calls it too). Hermes honors a
    # transport its config keeps for a custom endpoint, so one left from an earlier model (Anthropic's Messages,
    # for example) would send this endpoint requests in another API.
    as_hermes hermes config set model.api_mode chat_completions
    if [ -n "${COLONYMIND_CONTAINER_MODEL_KEY:-}" ]; then
        # Single quotes: the reference itself goes into Hermes' config, which reads the value from the environment.
        # shellcheck disable=SC2016
        as_hermes hermes config set model.api_key '${COLONYMIND_CONTAINER_MODEL_KEY}'
    fi
    say "Hermes' model is $model at $model_url (COLONYMIND_CONTAINER_MODEL_URL); the sidecar uses the same endpoint"
    # Init keeps the endpoint it recorded unless it is told one: tell it this one (an explicit answer still wins).
    export COLONYMIND_INIT_MODEL_URL="${COLONYMIND_INIT_MODEL_URL:-$model_url}"
    export COLONYMIND_INIT_MODEL="${COLONYMIND_INIT_MODEL:-$model}"
    # Init selects the router's key on its own: its override, else the key Hermes' config resolves to, which falls
    # back to OPENAI_API_KEY in Hermes' .env. That key was given for no endpoint here (nor is Hermes' own, when
    # ColonyMind is told another endpoint than Hermes'): when the endpoint has no key of its own, init is told so,
    # and neither writes another's key into a router file it creates nor sends it with its recall probe.
    verdict="$(as_hermes "$python" "$endpoint" init-key "$hermes_home" "$COLONYMIND_INIT_MODEL_URL")" || {
        say "cannot tell which key init would give $COLONYMIND_INIT_MODEL_URL (above); the sidecar starts once it can"
        exit 1
    }
    case "$verdict" in
        same) ;;
        keyless) export COLONYMIND_MODEL_API_KEY=local-no-key;;
        *)
            say "unexpected answer from $endpoint: $verdict"
            exit 1;;
    esac
    # The router reads a file of its own, which init writes only when it is missing. The helper points it at this
    # endpoint: its URL, its key (the router's own override first, else the one given for this endpoint, or none)
    # and every tier's model, keeping everything else the file says. It is rewritten only when that changes
    # something, with the version it replaces in backups/: in this container the variables are the answer. A file
    # it cannot read stops setup instead of being taken for a changed endpoint.
    note="$(as_hermes "$python" "$endpoint" router "$home/.colonymind-llm-config.json" "$hermes_home" \
        "$COLONYMIND_INIT_MODEL_URL" "$COLONYMIND_INIT_MODEL")" || {
        say "the router's file is left as it is (the reason is above); the sidecar starts once setup can finish"
        exit 1
    }
    if [ -n "$note" ]; then
        say "$note"
    fi
fi

if [ -f "$home/colonymind.yaml" ]; then
    as_hermes colonymind upgrade
elif [ -z "${COLONYMIND_INIT_OWNER_HANDLES:-}" ]; then
    say "no owner handles (COLONYMIND_INIT_OWNER_HANDLES, PLATFORM=ID separated by commas): only the terminal" \
        "inside this container counts as the owner, and the owner's messages from any other platform are refused"
fi
as_hermes colonymind init --non-interactive
