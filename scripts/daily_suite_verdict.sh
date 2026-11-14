#!/usr/bin/env bash
# Append one dated line per start; keep the full output of the three newest runs beside the suite artifacts.
set -euo pipefail
# An exported BASHOPTS can turn these on; with them names of another case would be listed and taken for own output.
shopt -u nocasematch nocaseglob
umask 077
if [ "$#" -ne 2 ]; then
  echo "usage: $0 CHECKOUT VERDICT_LOG" >&2
  exit 2
fi
log="$2"
commit='?'; output='?'; known='?'; total='?'; new='?'; passing='?'; verdict=INCOMPLETE; reason=''; unremoved=''

# The one line of this start: in the verdict log, or on standard error when the log cannot be written.
record() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$commit verdict=$verdict"
  if [ "$verdict" = INCOMPLETE ]; then line+=" reason=\"$reason\""; fi
  line+=" known=$known/$total new=$new now_passing=$passing"
  if [ -n "$unremoved" ]; then line+=" not_removed=$unremoved"; fi
  line+=" output=$output"
  if { printf '%s\n' "$line" >> "$log"; } 2> /dev/null; then
    printf '%s\n' "$line"
  else
    printf '%s\n' "$line" >&2
    return 1
  fi
}

# A start that cannot run says why and ends with the status it always had.
refuse() {
  reason="$2"
  record || true
  exit "$1"
}

# Output of the runner or of this wrapper, by name and kind; never a symbolic link, never the verdict log's file.
# The name is tested in the C locale: six ASCII letters or digits, whatever locale this wrapper starts in.
own_output() {
  local name="${1##*/}" LC_ALL=C
  if [ -L "$1" ] || [ "$1" -ef "$log" ]; then return 1; fi
  if [[ "$name" =~ ^colonymind-suite\.[A-Za-z0-9]{6}$ ]]; then [ -d "$1" ]
  elif [[ "$name" =~ ^daily-suite\.[A-Za-z0-9]{6}\.log$ ]]; then [ -f "$1" ]
  else return 1
  fi
}

# Of the entries named, remove those that earlier runs left and that three others are newer than.
prune() {
  local entry other newer
  for entry in "$@"; do
    if [[ "$earlier" != *"/${entry##*/}/"* ]] || ! own_output "$entry"; then continue; fi
    newer=0
    for other in "$@"; do
      if own_output "$other" && [ "$other" -nt "$entry" ]; then newer=$((newer + 1)); fi
    done
    if [ "$newer" -ge 3 ] && ! rm -rf -- "$entry"; then unremoved+="${unremoved:+,}${entry##*/}"; fi
  done
}

# Stopped by the service manager: leave the line, then end as the signal would have ended the run.
stopped() {
  trap '' TERM
  record || true
  trap - TERM
  kill -TERM "$$"
}

checkout="$(cd "$1" && pwd)" || refuse "$?" 'the checkout path cannot be entered'
[ -n "${SUITE_PRIVATE_ROOT:-}" ] || refuse 1 'SUITE_PRIVATE_ROOT is not set to a private output directory'
commit="$(git -C "$checkout" rev-parse HEAD)" \
  || { status=$?; commit='?'; refuse "$status" 'git finds no commit in the checkout'; }
[ -d "$SUITE_PRIVATE_ROOT" ] || refuse 1 'SUITE_PRIVATE_ROOT is not a directory'
# What earlier runs left; nothing that appears from here on is ever removed by this run.
earlier='/'
for entry in "$SUITE_PRIVATE_ROOT"/colonymind-suite.?????? "$SUITE_PRIVATE_ROOT"/daily-suite.??????.log; do
  if own_output "$entry"; then earlier+="${entry##*/}/"; fi
done
output="$(mktemp "$SUITE_PRIVATE_ROOT/daily-suite.XXXXXX.log")" \
  || { status=$?; output='?'; refuse "$status" 'mktemp made no output file in SUITE_PRIVATE_ROOT'; }
reason='stopped by SIGTERM before the runner ended'
trap stopped TERM
status=0
bash "$checkout/scripts/suite_env.sh" run > "$output" 2>&1 || status=$?
reason="the runner ended with status $status and without one final verdict line"
last="$(tail -n 1 "$output")"
pattern='^== known failures: ([0-9]+) of ([0-9]+) still fail; ([0-9]+) new; ([0-9]+) now pass$'
if [[ "$last" =~ $pattern ]] && [ "$(grep -c '^== known failures:' "$output")" -eq 1 ]; then
  known="${BASH_REMATCH[1]}"; total="${BASH_REMATCH[2]}"
  new="${BASH_REMATCH[3]}"; passing="${BASH_REMATCH[4]}"
  reason="the runner's status $status does not fit its verdict line"
  if [ "$status" -eq 0 ] && [ "$new" -eq 0 ]; then
    verdict=PASS
    [ "$known" -eq 0 ] || verdict=KNOWN_FAILURES
  elif [ "$status" -eq 1 ] && [ "$new" -gt 0 ]; then
    verdict=NEW_FAILURES
  fi
fi
if [ "$verdict" = INCOMPLETE ]; then status=2; fi
# The three newest run directories and the three newest logs of this wrapper stay.
prune "$SUITE_PRIVATE_ROOT"/colonymind-suite.??????
prune "$SUITE_PRIVATE_ROOT"/daily-suite.??????.log
trap '' TERM
record || exit 1
exit "$status"
