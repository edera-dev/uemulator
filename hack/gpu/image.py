#!/usr/bin/env python3
"""Builds a guest that draws through Mesa's virgl driver, the check for M0 of
docs/plans/25-gpu-3d.md, into OUT (target/gpu):

- vmlinuz-virt, Alpine's virt kernel (demo.py's, pinned there);
- initramfs.gz: Alpine's minirootfs, the virtio-gpu, virtio-blk and squashfs modules, and an
  /init that mounts the disk, runs eglclear in the background and starts a shell in the disk;
- rootfs.squashfs, the disk: the minirootfs with Mesa's virgl driver, EGL, GBM and GLES,
  kmscube, and eglclear (hack/gpu/eglclear.c). It is a disk and not part of the initramfs
  because Mesa needs LLVM, a quarter of a gigabyte the guest then reads only as it touches it.

    hack/gpu/image.py [OUT] [--cache DIR]

Then, under Node (fences, the decoded clears and the scanout on stderr):

    node hack/wasm/boot.mjs --gpu --kernel target/gpu/vmlinuz-virt --initrd target/gpu/initramfs.gz \\
        --disk target/gpu/rootfs.squashfs --memory 1024 --cmdline eglclear.hold=0 --until "EGLCLEAR done"

or on the page with ?gpu=1, the same three files and the Screen view: red, green, blue, red.
`eglclear.hold=N` on the command line holds each colour N seconds (3 by default);
`gpu.run=PROGRAM` runs PROGRAM (`kmscube`, say: page flips with a fence each frame) instead.

Needs mksquashfs 4.6 or later (it reads the tree as a tar on stdin, so nothing is unpacked onto
this machine's file system but the headers eglclear compiles against) and x86_64-linux-musl-gcc
(CC overrides). The packages are what Alpine's edge has today, fetched over HTTPS without
checking APKINDEX's signature: a check to run locally, not something published."""
import argparse
import gzip
import io
import os
import re
import stat
import subprocess
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web"))
import demo  # noqa: E402

REPOSITORIES = ["main", "community"]
# Edge, not demo.py's release: Alpine's stable branches up to 3.24 build Mesa without virgl
# (`_gallium_drivers` in aports' main/mesa/APKBUILD), so Mesa's loader prints "virtio_gpu: driver
# missing" and falls back to llvmpipe. Edge's packages over the release's minirootfs and kernel.
PACKAGE_BRANCH = "edge"
MIRROR = f"https://dl-cdn.alpinelinux.org/alpine/{PACKAGE_BRANCH}"
# Mesa's virgl driver is in libgallium (the `mesa` package, which these pull in).
PACKAGES = ["mesa-dri-gallium", "mesa-egl", "mesa-gbm", "mesa-gles", "kmscube"]
# Only their headers and link names, for compiling eglclear.
DEVELOPMENT_PACKAGES = ["mesa-dev", "libdrm-dev"]
# What /init loads; their dependencies come from the kernel's modules.dep.
MODULES = ["kernel/drivers/gpu/drm/virtio/virtio-gpu.ko", "kernel/drivers/block/virtio_blk.ko",
           "kernel/fs/squashfs/squashfs.ko"]
MODULE_DIR = "lib/modules/gpu"

INIT = f"""#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev
tty=$(cat /sys/class/tty/console/active)
tty=${{tty##* }}
exec <>"/dev/$tty" >&0 2>&1
for module in @MODULES@; do
    insmod /{MODULE_DIR}/$module.ko || echo "insmod $module failed"
done
dmesg | grep -i "virgl\\|virtio_gpu\\|\\[drm\\]"
mkdir -p /disk
mount -t squashfs -o ro /dev/vda /disk || {{ echo "no rootfs.squashfs on vda"; exec sh; }}
mount -t proc proc /disk/proc
mount -t sysfs sys /disk/sys
mount -o bind /dev /disk/dev
mount -t tmpfs tmpfs /disk/tmp
mount -t tmpfs tmpfs /disk/run
hold=3
run=
for argument in $(cat /proc/cmdline); do
    case "$argument" in
        eglclear.hold=*) hold=${{argument#eglclear.hold=}} ;;
        gpu.run=*) run=${{argument#gpu.run=}} ;;
    esac
done
echo "eglclear clears the screen red, green, blue and red through virgl; 'kmscube' draws a cube."
# kmscube quits as soon as its stdin reads ("user interrupted!"); a pipe nothing writes to never does.
sleep 2147483647 | chroot /disk ${{run:-/usr/bin/eglclear $hold}} &
export HOME=/root
exec chroot /disk setsid -c /bin/sh -l
"""


def fetch(url, cache):
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(path):
        print(f"gpu: downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        with open(path + ".part", "wb") as out:
            out.write(data)
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        return f.read()


def index(cache):
    """{name: fields} and {provided name: package} from both repositories' APKINDEX."""
    packages, provides = {}, {}
    for repository in REPOSITORIES:
        # Never cached: the packages it names are replaced as the branch moves on.
        url = f"{MIRROR}/{repository}/x86_64/APKINDEX.tar.gz"
        print(f"gpu: downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            text = tar.extractfile("APKINDEX").read().decode()
        for block in text.split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if len(line) > 2 and line[1] == ":":
                    fields.setdefault(line[0], line[2:])
            if "P" not in fields:
                continue
            fields["repository"] = repository
            packages.setdefault(fields["P"], fields)
            for provided in fields.get("p", "").split():
                provides.setdefault(re.split(r"[=<>~]", provided)[0], fields["P"])
    return packages, provides


def closure(names, packages, provides):
    """Every package `names` need, by name."""
    seen, todo = set(), list(names)
    while todo:
        dependency = re.split(r"[=<>~]", todo.pop())[0]
        if dependency.startswith("!"):
            continue
        name = dependency if dependency in packages else provides.get(dependency)
        if name is None:
            sys.exit(f"gpu: nothing in {PACKAGE_BRANCH} provides {dependency}")
        if name not in seen:
            seen.add(name)
            todo.extend(packages[name].get("D", "").split())
    return sorted(seen)


def apk(name, packages, cache):
    fields = packages[name]
    return fetch(f"{MIRROR}/{fields['repository']}/x86_64/{name}-{fields['V']}.apk", cache)


def apk_members(data):
    """(TarInfo, bytes or None) of a package's files, without its signature and metadata. An apk
    is gzip streams back to back, each a tar with its own end marker."""
    with tarfile.open(fileobj=gzip.GzipFile(fileobj=io.BytesIO(data)), mode="r|", ignore_zeros=True) as tar:
        for member in tar:
            if member.name.startswith("."):
                continue
            yield member, tar.extractfile(member).read() if member.isfile() else None


def module_order(netboot_initramfs_gz):
    """MODULES and what they need, below usr/lib/modules/VERSION/, each after its dependencies."""
    dependencies = {}
    for name, data in demo.newc_members(gzip.decompress(netboot_initramfs_gz)):
        if re.fullmatch(r"usr/lib/modules/[^/]+/modules\.dep", name):
            for line in data.decode().splitlines():
                module, _, needs = line.partition(":")
                dependencies[module] = needs.split()
    order = []

    def visit(module):
        if module in order:
            return
        if module not in dependencies:
            sys.exit(f"gpu: the netboot initramfs has no {module}")
        for needed in reversed(dependencies[module]):
            visit(needed)
        order.append(module)

    for module in MODULES:
        visit(module)
    return order


def safe_name(name):
    name = name.removeprefix("./").rstrip("/")
    if name.startswith("/") or ".." in name.split("/"):
        sys.exit(f"gpu: refusing the path {name}")
    return name


def compile_eglclear(development, runtime, cache):
    """eglclear, linked against the guest's own libraries. The headers and libraries are
    unpacked under the cache, every path checked to stay inside it."""
    sysroot = os.path.realpath(os.path.join(cache, "sysroot"))
    for data in development + runtime:
        for member, content in apk_members(data):
            name = safe_name(member.name)
            if not name.startswith(("usr/include/", "usr/lib/", "lib/")) or member.isdir():
                continue
            path = os.path.join(sysroot, name)
            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)
            # A symlinked directory in a package could otherwise point a later file outside.
            if not os.path.realpath(parent).startswith(sysroot + os.sep):
                sys.exit(f"gpu: {name} would land outside {sysroot}")
            if os.path.lexists(path):
                os.remove(path)
            if member.issym():
                os.symlink(member.linkname, path)
            elif member.isfile():
                with open(path, "wb") as out:
                    out.write(content)
    out = os.path.join(cache, "eglclear")
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eglclear.c")
    compiler = os.environ.get("CC", "x86_64-linux-musl-gcc")
    subprocess.run([compiler, "-O2", "-Wall", "-DEGL_NO_X11", "-DMESA_EGL_NO_X11_HEADERS",
                    "-I", f"{sysroot}/usr/include", "-I", f"{sysroot}/usr/include/libdrm",
                    "-L", f"{sysroot}/usr/lib", f"-Wl,-rpath-link,{sysroot}/usr/lib",
                    f"-Wl,-rpath-link,{sysroot}/lib", "-o", out, source,
                    "-lEGL", "-lGLESv2", "-lgbm", "-ldrm"], check=True)
    with open(out, "rb") as f:
        return f.read()


def rootfs(rootfs_tar_gz, packages, eglclear, out_path):
    """The squashfs: the minirootfs, then every package over it (a later file replaces an
    earlier one, as apk would), then eglclear."""
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(rootfs_tar_gz), mode="r:gz") as tar:
        for member in tar:
            entries[safe_name(member.name)] = (member, tar.extractfile(member).read() if member.isfile() else None)
    for data in packages:
        for member, content in apk_members(data):
            name = safe_name(member.name)
            if member.isdir() and name in entries:
                continue
            if member.islnk():
                target = entries.get(safe_name(member.linkname))
                if not target or not target[0].isfile():
                    sys.exit(f"gpu: {name} is a hard link to {member.linkname}, which is not a file")
                member, content = target[0], target[1]
            entries[name] = (member, content)
    eglclear_info = tarfile.TarInfo("usr/bin/eglclear")
    eglclear_info.mode = 0o755
    entries["usr/bin/eglclear"] = (eglclear_info, eglclear)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, (member, content) in entries.items():
            if name in ("", ".") or not (member.isfile() or member.isdir() or member.issym()):
                continue
            info = tarfile.TarInfo(name)
            info.mode, info.uid, info.gid, info.mtime = member.mode, 0, 0, 0
            if member.isdir():
                info.type = tarfile.DIRTYPE
            elif member.issym():
                info.type, info.linkname = tarfile.SYMTYPE, member.linkname
            else:
                info.size = len(content)
            tar.addfile(info, io.BytesIO(content) if content is not None else None)
    if os.path.exists(out_path):
        os.remove(out_path)
    # gzip because the virt kernel's squashfs has zlib built in.
    subprocess.run(["mksquashfs", "-", out_path, "-tar", "-all-root", "-comp", "gzip", "-quiet", "-no-progress"],
                   input=archive.getvalue(), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=os.path.join("target", "gpu"))
    parser.add_argument("--cache", default=os.path.join("target", "gpu-cache"))
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    kernel = demo.fetch(*demo.KERNEL, args.cache)
    minirootfs = demo.fetch(*demo.ROOTFS, args.cache)
    netboot = demo.fetch(*demo.NETBOOT_INITRAMFS, args.cache)
    order = module_order(netboot)
    modules = demo.net_modules(netboot, wanted=order)
    names = " ".join(os.path.basename(module).removesuffix(".ko") for module in order)
    initramfs = demo.initramfs(minirootfs, modules, init=INIT.replace("@MODULES@", names), module_dir=MODULE_DIR)

    packages, provides = index(args.cache)
    runtime_names = closure(PACKAGES, packages, provides)
    print(f"gpu: {len(runtime_names)} packages: {' '.join(runtime_names)}", file=sys.stderr)
    runtime = [apk(name, packages, args.cache) for name in runtime_names]
    development = [apk(name, packages, args.cache) for name in DEVELOPMENT_PACKAGES]
    eglclear = compile_eglclear(development, runtime, args.cache)

    with open(os.path.join(args.out, "vmlinuz-virt"), "wb") as out:
        out.write(kernel)
    with open(os.path.join(args.out, "initramfs.gz"), "wb") as out:
        out.write(initramfs)
    rootfs(minirootfs, runtime, eglclear, os.path.join(args.out, "rootfs.squashfs"))
    for name in ("vmlinuz-virt", "initramfs.gz", "rootfs.squashfs"):
        size = os.path.getsize(os.path.join(args.out, name))
        print(f"gpu: {args.out}/{name}, {size / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
