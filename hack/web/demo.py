#!/usr/bin/env python3
"""Builds the web page's demo machine into OUT (target/web/demo): Alpine's virt kernel, an
initramfs made from Alpine's minirootfs with an /init that starts a shell on the console, and
demo.json, which puts the page in demo mode (one button that boots it). `WEB_DEMO=1 hack/web.sh`
runs it, and the Pages workflow sets that.

    hack/web/demo.py OUT [--cache DIR]

The downloads are pinned by version and SHA-256; a mismatch stops the build. To move to a new
Alpine release, change ALPINE_VERSION and both hashes (latest-releases.yaml next to the
minirootfs lists that one; hash the kernel yourself). Only the standard library: the tarball is
read, never extracted, so nothing in it can write outside OUT."""
import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import urllib.request

ALPINE_VERSION = "3.24.3"
ALPINE_BRANCH = "v" + ".".join(ALPINE_VERSION.split(".")[:2])
RELEASES = f"https://dl-cdn.alpinelinux.org/alpine/{ALPINE_BRANCH}/releases/x86_64"
KERNEL = (f"{RELEASES}/netboot-{ALPINE_VERSION}/vmlinuz-virt",
          "17fbd3efd16f2e7cb1b945f06423b37399ba7d5ac8cdd3665e7f13f363500e85")
ROOTFS = (f"{RELEASES}/alpine-minirootfs-{ALPINE_VERSION}-x86_64.tar.gz",
          "f8e1cafba9b1b283b5d30d06f9885c8d2bc161ffe9083ade8e2e157813a53407")

# PID 1. The minirootfs has busybox but no OpenRC, so its own /sbin/init would fail on inittab.
# The shell gets the console's real tty (the last entry of console/active: ttyS0 or hvc0) as its
# controlling terminal; /dev/console cannot be one, and without it there is no job control and
# no Ctrl-C. `poweroff -f` because nothing here would act on the signal plain poweroff sends.
INIT = f"""#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev
mkdir -p /dev/pts /dev/shm
mount -t devpts devpts /dev/pts
mount -t tmpfs tmpfs /tmp
hostname uemulator
tty=$(cat /sys/class/tty/console/active)
tty=${{tty##* }}
exec <>"/dev/$tty" >&0 2>&1
echo
echo "Alpine Linux {ALPINE_VERSION} on uemulator, running from RAM in your browser."
echo "No network. 'poweroff -f' ends the machine."
echo
export HOME=/root
cd /root
exec setsid -c /bin/sh -l
"""

NEWC_MAGIC = b"070701"
NEWC_TRAILER = "TRAILER!!!"
CPIO_ALIGN = 4
CONSOLE_MAJOR, CONSOLE_MINOR = 5, 1


def fetch(url, sha256, cache):
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(path):
        print(f"demo: downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        with open(path + ".part", "wb") as out:
            out.write(data)
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        os.remove(path)
        sys.exit(f"demo: {url} has SHA-256 {actual}, expected {sha256}")
    return data


class Newc:
    """A cpio archive in the newc format the kernel unpacks, owned by root, every mtime 0, so the
    same inputs give the same bytes."""

    def __init__(self):
        self.out = io.BytesIO()
        self.inode = 1

    def _pad(self):
        self.out.write(b"\0" * (-self.out.tell() % CPIO_ALIGN))

    def add(self, name, mode, data=b"", rdev=(0, 0)):
        encoded = name.encode() + b"\0"
        fields = [self.inode, mode, 0, 0, 2 if stat.S_ISDIR(mode) else 1, 0, len(data),
                  0, 0, rdev[0], rdev[1], len(encoded), 0]
        self.inode += 1
        self.out.write(NEWC_MAGIC + b"".join(b"%08x" % field for field in fields))
        self.out.write(encoded)
        self._pad()
        self.out.write(data)
        self._pad()

    def finish(self):
        self.add(NEWC_TRAILER, 0)
        return self.out.getvalue()


def initramfs(rootfs_tar_gz):
    archive = Newc()
    with tarfile.open(fileobj=io.BytesIO(rootfs_tar_gz), mode="r:gz") as tar:
        for member in tar:
            name = member.name.removeprefix("./").rstrip("/")
            if not name or name == "." or name == "init":
                continue
            permissions = member.mode & 0o7777
            if member.isdir():
                archive.add(name, stat.S_IFDIR | permissions)
            elif member.isfile():
                archive.add(name, stat.S_IFREG | permissions, tar.extractfile(member).read())
            elif member.issym():
                archive.add(name, stat.S_IFLNK | 0o777, member.linkname.encode())
            else:
                sys.exit(f"demo: {member.name} in the minirootfs is neither a file, directory nor symlink")
    # Where the kernel opens PID 1's first stdin/stdout, before devtmpfs is mounted over /dev.
    archive.add("dev/console", stat.S_IFCHR | 0o600, rdev=(CONSOLE_MAJOR, CONSOLE_MINOR))
    archive.add("init", stat.S_IFREG | 0o755, INIT.encode())
    return gzip.compress(archive.finish(), compresslevel=9, mtime=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--cache", default=os.path.join("target", "web-demo-cache"))
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    kernel = fetch(*KERNEL, args.cache)
    rootfs = fetch(*ROOTFS, args.cache)
    files = {"vmlinuz-virt": kernel, "initramfs.gz": initramfs(rootfs)}
    for name, data in files.items():
        with open(os.path.join(args.out, name), "wb") as out:
            out.write(data)
    # What main.js boots in demo mode; paths are relative to this file.
    demo = {
        # `label` is the start dialog's button ("Boot Alpine Linux"), `name` the status bar's.
        "label": "Alpine Linux",
        "name": f"Alpine Linux {ALPINE_VERSION}",
        "kernel": "vmlinuz-virt",
        "initrd": "initramfs.gz",
        "firmware": "uefi",
        "console": "serial",
        "memoryMib": 256,
        "bytes": sum(len(data) for data in files.values()),
    }
    with open(os.path.join(args.out, "demo.json"), "w") as out:
        json.dump(demo, out, indent=2)
        out.write("\n")
    print(f"demo: {demo['name']}, {demo['bytes'] / 1e6:.1f} MB in {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
