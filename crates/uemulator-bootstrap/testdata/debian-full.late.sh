#!/bin/sh
# uemulator quick bootstrap: the Debian installer's preseed/late_command runs this once the
# install has succeeded, with the new system mounted at /target.
set -e
mkdir -p /target/home/alex/.ssh
printf '%s\n' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyOne alex@laptop' 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABFakeKeyTwo it'\''s "mine"' > /target/home/alex/.ssh/authorized_keys
chmod 700 /target/home/alex/.ssh
chmod 600 /target/home/alex/.ssh/authorized_keys
in-target chown -R alex:alex /home/alex/.ssh
printf '%s: %s\n' uemulator-bootstrap installed > /dev/console
