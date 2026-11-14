<!--
Release notes template. For a release, copy everything below this comment to the top of CHANGELOG.md, under
"# Changelog", and fill it in. The Release workflow publishes that section as the GitHub release: the heading after
"## " becomes the release title, the rest becomes its notes.

- Keep the heading shape "## vX.Y.Z - title", with a plain hyphen.
- Write for the person who runs ColonyMind: what they will notice, in plain words. No internal unit names,
  branch names or private measurements.
- Keep "Upgrading" even when nothing extra is needed; say so there.
- Delete a section that has nothing in it, and every placeholder line.
-->

## vX.Y.Z - what this release changes, in a few plain words

Two or three sentences: what changes for the person running ColonyMind, and why it matters to them.

### What changes

- A change someone will notice or can rely on, described by what it does for them.

### Fixes

- A problem that is gone, described by its symptom.

### Upgrading

    uv tool upgrade colonymind
    colonymind upgrade
    hermes gateway restart

- What `colonymind upgrade` changes on disk or in configuration, and anything to do by hand; or "Nothing else to do."
- Supported Hermes and Python versions, if they changed (keep docs/COMPATIBILITY.md in step).
