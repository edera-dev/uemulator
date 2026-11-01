#!/usr/bin/env python3
"""Builds the web page's demo machines into OUT (target/web/demo): Alpine's virt kernel, an
initramfs made from Alpine's minirootfs with an /init that starts a shell on the console, Edera's
Xen (xen.gz), and demo.json, which puts the page in demo mode: a button per machine, "Alpine
Linux" (the kernel and initramfs) and "Edera Xen" (the same pair as Xen's dom0). `WEB_DEMO=1
hack/web.sh` runs it, and the Pages workflow sets that.

    hack/web/demo.py OUT [--cache DIR]

The downloads are pinned; a mismatch stops the build. Alpine by version and SHA-256: to move to
a new release, change ALPINE_VERSION and both hashes (latest-releases.yaml next to the
minirootfs lists that one; hash the kernel yourself). Xen by the image manifest digest in
demo-xen.ref, which covers every layer: the registry's answers are checked against it. Only the
standard library: tarballs are read, never extracted, so nothing in them can write outside OUT."""
import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import urllib.parse
import urllib.request

ALPINE_VERSION = "3.24.3"
ALPINE_BRANCH = "v" + ".".join(ALPINE_VERSION.split(".")[:2])
RELEASES = f"https://dl-cdn.alpinelinux.org/alpine/{ALPINE_BRANCH}/releases/x86_64"
KERNEL = (f"{RELEASES}/netboot-{ALPINE_VERSION}/vmlinuz-virt",
          "17fbd3efd16f2e7cb1b945f06423b37399ba7d5ac8cdd3665e7f13f363500e85")
ROOTFS = (f"{RELEASES}/alpine-minirootfs-{ALPINE_VERSION}-x86_64.tar.gz",
          "f8e1cafba9b1b283b5d30d06f9885c8d2bc161ffe9083ade8e2e157813a53407")

XEN_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo-xen.ref")
# Where Edera's image keeps them.
XEN_IMAGE_PATH = "boot/xen.gz"
XEN_CONFIG_PATH = "boot/xen.config"
OCI_MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])
GZIP_LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
}
# Xen keeps the 16550 and passes dom0's console (hvc0) through it. Left to itself Xen gives dom0
# a share of the memory that is too small for this initramfs at this size, so it is fixed.
# `dom0=pvh-auto` (Edera's; upstream has only pv and pvh) builds a PVH dom0 when Xen has HVM and
# an IOMMU and falls back to PV otherwise, so the machine has the IOMMU: without it, or with the
# page's IOMMU switch turned off, the same demo still boots, as a PV dom0.
XEN_MEMORY_MIB = 512
XEN_ARGS = "console=com1 com1=115200,8n1 dom0_mem=384M,max:384M dom0=pvh-auto"

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
if [ "$(cat /sys/hypervisor/type 2>/dev/null)" = xen ]; then
    v=/sys/hypervisor/version
    echo "It is the $(cat /sys/hypervisor/guest_type) dom0 of Xen $(cat $v/major).$(cat $v/minor)$(cat $v/extra); see 'dmesg' and /sys/hypervisor."
fi
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


def read_verified(path, sha256):
    """The file's bytes if they hash to `sha256`, else None (and the file is gone)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() == sha256:
        return data
    os.remove(path)
    return None


class Registry:
    """Pulls by digest from an OCI registry (ghcr.io) with its anonymous pull token, fetched
    only when something is not cached. Everything is checked against its digest, cached or not:
    neither the registry nor the CDN it redirects blobs to is trusted."""

    def __init__(self, registry, repository, cache):
        self.registry, self.repository, self.cache = registry, repository, cache
        self.token = None

    def _token(self):
        if self.token is None:
            scope = urllib.parse.quote(f"repository:{self.repository}:pull")
            with urllib.request.urlopen(f"https://{self.registry}/token?scope={scope}") as response:
                self.token = json.load(response)["token"]
        return self.token

    def get(self, digest, accept=None):
        """A manifest (with `accept`) or a blob."""
        algorithm, _, sha256 = digest.partition(":")
        if algorithm != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            sys.exit(f"demo: unsupported digest {digest}")
        path = os.path.join(self.cache, f"oci-{sha256}")
        data = read_verified(path, sha256)
        if data is not None:
            return data
        kind = "manifests" if accept else "blobs"
        url = f"https://{self.registry}/v2/{self.repository}/{kind}/{digest}"
        print(f"demo: downloading {url}", file=sys.stderr)
        request = urllib.request.Request(url)
        if accept:
            request.add_header("Accept", accept)
        # Unredirected: a blob redirects to a signed CDN URL, which must not see the token.
        request.add_unredirected_header("Authorization", f"Bearer {self._token()}")
        with urllib.request.urlopen(request) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            sys.exit(f"demo: {url} has SHA-256 {actual}, expected {sha256}")
        with open(path + ".part", "wb") as out:
            out.write(data)
        os.replace(path + ".part", path)
        return data


def fetch_xen(cache):
    """xen.gz and the Xen version from the image demo-xen.ref pins. Later layers win, as in the
    image's file system."""
    with open(XEN_REF) as f:
        ref = next(line.strip() for line in f if line.strip() and not line.startswith("#"))
    match = re.fullmatch(r"([^/]+)/(.+)@(sha256:[0-9a-f]{64})", ref)
    if not match:
        sys.exit(f"demo: {XEN_REF}: expected REGISTRY/REPOSITORY@sha256:DIGEST, got {ref}")
    registry, repository, digest = match.groups()
    pull = Registry(registry, repository, cache)
    manifest = json.loads(pull.get(digest, accept=OCI_MANIFEST_TYPES))
    wanted = (XEN_IMAGE_PATH, XEN_CONFIG_PATH)
    whiteouts = {f"{os.path.dirname(path)}/.wh.{os.path.basename(path)}": path for path in wanted}
    found = {}
    for layer in manifest["layers"]:
        if layer["mediaType"] not in GZIP_LAYER_TYPES:
            sys.exit(f"demo: {ref} has a {layer['mediaType']} layer; only gzip layers can be read")
        with tarfile.open(fileobj=io.BytesIO(pull.get(layer["digest"])), mode="r:gz") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if name in wanted and member.isfile():
                    found[name] = tar.extractfile(member).read()
                elif name in whiteouts:
                    found.pop(whiteouts[name], None)
    if XEN_IMAGE_PATH not in found:
        sys.exit(f"demo: {ref} has no {XEN_IMAGE_PATH}")
    # "# Xen/x86 4.22.1-pre Configuration" heads the build's Kconfig.
    version = re.search(rb"Xen/x86 (\S+) Configuration", found.get(XEN_CONFIG_PATH, b""))
    return found[XEN_IMAGE_PATH], version.group(1).decode() if version else None


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
    xen, xen_version = fetch_xen(args.cache)
    files = {"vmlinuz-virt": kernel, "initramfs.gz": initramfs(rootfs), "xen.gz": xen}
    for name, data in files.items():
        with open(os.path.join(args.out, name), "wb") as out:
            out.write(data)
    alpine_bytes = len(files["vmlinuz-virt"]) + len(files["initramfs.gz"])
    # What main.js boots in demo mode; paths are relative to this file. The first is what
    # ?autoboot boots; ?demo=ID picks another. `label` is the welcome pane's button ("Boot Alpine
    # Linux") and the preset, `name` the status bar's.
    machines = [
        {
            "id": "alpine",
            "label": "Alpine Linux",
            "name": f"Alpine Linux {ALPINE_VERSION}",
            "kernel": "vmlinuz-virt",
            "initrd": "initramfs.gz",
            "firmware": "uefi",
            "console": "serial",
            "memoryMib": 256,
            "bytes": alpine_bytes,
        },
        {
            "id": "xen",
            "label": "Edera Xen",
            "name": f"Edera Xen {xen_version or ''} with an Alpine dom0".replace("  ", " "),
            "hypervisor": "xen.gz",
            "hypervisorArgs": XEN_ARGS,
            "kernel": "vmlinuz-virt",
            "initrd": "initramfs.gz",
            "firmware": "uefi",
            "console": "serial",
            "memoryMib": XEN_MEMORY_MIB,
            "iommu": True,
            "bytes": alpine_bytes + len(files["xen.gz"]),
        },
    ]
    with open(os.path.join(args.out, "demo.json"), "w") as out:
        json.dump({"machines": machines}, out, indent=2)
        out.write("\n")
    for machine in machines:
        print(f"demo: {machine['name']}, {machine['bytes'] / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
