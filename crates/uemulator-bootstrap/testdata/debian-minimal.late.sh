#!/bin/sh
# uemulator quick bootstrap: the Debian installer's preseed/late_command runs this once the
# install has succeeded, with the new system mounted at /target.
set -e
printf '%s: %s\n' uemulator-bootstrap installed > /dev/console
