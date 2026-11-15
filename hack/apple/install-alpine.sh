#!/bin/sh
# Installs Alpine Linux to /dev/vda from the Alpine demo (hack/web/demo.py) with its network on:
# the check that a machine in the Apple app installs a system and boots it after the app quits
# (docs/plans/28-apple.md, "Installing Alpine"). Typed or pasted into the demo's shell; needs about
# 1 GiB of RAM (the root is assembled in /staging, in RAM) and a disk of 1 GiB or more.
#
# The demo kernel loads almost no modules, so nothing here mounts a filesystem: `mkfs.ext4 -d`
# builds the root straight from /staging, mtools writes the EFI system partition, and the boot
# loader is a standalone GRUB at \EFI\BOOT\BOOTX64.EFI, which the uemulator firmware boots when
# NVRAM names nothing else. The installed system brings its own kernel, modules and initramfs.

fail() { echo "INSTALL FAILED: $1"; exit 1; }

apk add e2fsprogs dosfstools mtools sfdisk grub-efi >/dev/null || fail "tools"

# GPT: a 64 MiB EFI system partition, the rest Linux.
printf 'label: gpt\nsize=64M, type=U\ntype=L\n' | sfdisk -q /dev/vda || fail "partitions"
sleep 1
[ -b /dev/vda2 ] || fail "no /dev/vda2 after partitioning"

mkdir -p /staging/etc/apk
cp -r /etc/apk/keys /etc/apk/repositories /staging/etc/apk/
# A trigger (mkinitfs, under --root without /proc) may complain; the initramfs is made below.
apk --root /staging --initdb add alpine-base linux-virt mkinitfs >/dev/null
[ -f /staging/boot/vmlinuz-virt ] || fail "no kernel in /staging"
version=$(ls /staging/lib/modules)
mkinitfs -b /staging -o /staging/boot/initramfs-virt "$version" || fail "initramfs"

echo '/dev/vda2 / ext4 defaults 0 1' > /staging/etc/fstab
echo 'ttyS0::respawn:/sbin/getty -L 115200 ttyS0 vt100' >> /staging/etc/inittab
echo uemulator-installed > /staging/etc/hostname
mkfs.ext4 -q -F -d /staging /dev/vda2 || fail "root filesystem"

mkfs.vfat /dev/vda1 >/dev/null || fail "EFI system partition"
cat > /tmp/grub.cfg <<'GRUB'
set timeout=2
insmod part_gpt
insmod ext2
set root=(hd0,gpt2)
menuentry 'Alpine Linux (installed by uemulator)' {
  linux /boot/vmlinuz-virt root=/dev/vda2 rootfstype=ext4 modules=ext4 console=ttyS0
  initrd /boot/initramfs-virt
}
GRUB
grub-mkstandalone -O x86_64-efi -o /tmp/BOOTX64.EFI boot/grub/grub.cfg=/tmp/grub.cfg || fail "GRUB"
mmd -i /dev/vda1 ::/EFI ::/EFI/BOOT && mcopy -i /dev/vda1 /tmp/BOOTX64.EFI ::/EFI/BOOT/ || fail "boot loader"
sync
echo "INSTALLED $version"
