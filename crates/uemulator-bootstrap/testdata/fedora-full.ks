# uemulator quick bootstrap: Anaconda reads this from /ks.cfg in the initramfs
# (inst.ks=file:/ks.cfg on the kernel command line).
text
lang de_DE.UTF-8
keyboard --vckeymap=de
timezone Europe/Berlin --utc
network --bootproto=dhcp --hostname=devbox
url --url=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/
repo --name=everything --baseurl=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Everything/x86_64/os/
user --name=alex --groups=wheel --iscrypted --password=$6$testsalt$fakeuserhash
rootpw --iscrypted $6$testsalt$fakeroothash
sshkey --username=alex "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyOne alex@laptop"
sshkey --username=alex "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABFakeKeyTwo it's \"mine\""
services --enabled=sshd
firstboot --disable
ignoredisk --only-use=vda
zerombr
clearpart --all --initlabel --drives=vda
autopart
reboot

%packages
openssh-server
git
htop
%end

%post --nochroot
printf '%s: %s\n' uemulator-bootstrap installed > /dev/console
%end
