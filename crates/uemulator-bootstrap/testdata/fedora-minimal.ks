# uemulator quick bootstrap: Anaconda reads this from /ks.cfg in the initramfs
# (inst.ks=file:/ks.cfg on the kernel command line).
text
lang en_US.UTF-8
keyboard --vckeymap=us
timezone UTC --utc
network --bootproto=dhcp --hostname=box
url --url=https://dl.fedoraproject.org/pub/fedora/linux/releases/44/Server/x86_64/os/
user --name=alex --groups=wheel --iscrypted --password=$6$testsalt$fakeuserhash
rootpw --lock
firstboot --disable
ignoredisk --only-use=vda
zerombr
clearpart --all --initlabel --drives=vda
autopart
reboot

%packages
%end

%post --nochroot
printf '%s: %s\n' uemulator-bootstrap installed > /dev/console
%end
