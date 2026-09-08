#!/bin/sh
# Container entrypoint.
#
# The application must not run as root, but its storage directories are
# Docker named volumes. Docker only seeds a volume's ownership from the
# image when the volume is EMPTY - an existing volume from a previous
# deployment (or from before this image ran unprivileged) keeps whatever
# ownership it already had, which is typically root. The container then
# starts, passes its healthcheck, and fails on the first attempt to write
# a recording.
#
# So: start as root, take ownership of the mounted storage paths (cheap
# and idempotent), then drop to the unprivileged user for the actual
# server process. This is the same pattern the official postgres and
# nginx images use. `setpriv` comes from util-linux, already present in
# the base image - no extra package for this.
#
# If the container is already running as a non-root user (e.g. compose
# `user:` override, or a platform that enforces it), the chown is skipped
# and exec happens directly.
set -e

APP_USER=sentinelcam
APP_UID=10001
APP_GID=10001
STORAGE_DIR=/app/storage

if [ "$(id -u)" = "0" ]; then
    for dir in "$STORAGE_DIR/recordings" "$STORAGE_DIR/snapshots" "$STORAGE_DIR/uploads"; do
        mkdir -p "$dir"
        # -R because a volume that already holds recordings has
        # root-owned files inside it, not just a root-owned mount point.
        chown -R "$APP_UID:$APP_GID" "$dir" 2>/dev/null || \
            echo "entrypoint: warning - could not take ownership of $dir; writes may fail" >&2
    done

    exec setpriv --reuid "$APP_UID" --regid "$APP_GID" --init-groups --inh-caps=-all -- "$@"
fi

echo "entrypoint: already running as uid $(id -u); skipping ownership fixup" >&2
exec "$@"
