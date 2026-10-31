/*
 * Guest-side microbenchmarks for a Linux guest: syscalls, memcpy/memset and rep movs/stos,
 * O_DIRECT disk I/O, TCP to the host. Static, so it runs on any distro's userspace:
 *
 *   x86_64-linux-musl-gcc -O2 -static -o target/perf-guest/ubench hack/perf/guest/ubench.c
 *
 * `ubench [all|sys|mem|disk|net|randread] [path or host]`; `net` needs `netsrv.py` on the host.
 * Every line is `name value unit`, which `ab.py` collects. Times are the guest's own clock, which
 * follows the host's on the realtime clock (the default).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void report(const char *name, double v, const char *unit) { printf("%-24s %12.2f %s\n", name, v, unit); fflush(stdout); }

static void bench_syscall(long n) {
    double t = now();
    for (long i = 0; i < n; i++) syscall(SYS_getppid);
    report("syscall getppid", n / (now() - t), "calls/s");
    int z = open("/dev/zero", O_RDONLY), nul = open("/dev/null", O_WRONLY); char b[1];
    t = now();
    for (long i = 0; i < n / 2; i++) { read(z, b, 1); write(nul, b, 1); }
    report("syscall read+write 1B", n / (now() - t), "calls/s");
    t = now();
    for (long i = 0; i < n / 4; i++) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); }
    report("vdso clock_gettime", n / 4 / (now() - t), "calls/s");
}

static void __attribute__((noinline)) rep_movsb(void *d, const void *s, size_t n) { __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(n) :: "memory"); }
static void __attribute__((noinline)) rep_stosb(void *d, int c, size_t n) { __asm__ volatile("rep stosb" : "+D"(d), "+c"(n) : "a"(c) : "memory"); }

static void bench_mem(size_t size, int reps, const char *tag) {
    char *a = aligned_alloc(4096, size), *b = aligned_alloc(4096, size);
    memset(a, 1, size); memset(b, 2, size);
    char name[64]; double t;
    t = now(); for (int i = 0; i < reps; i++) { memcpy(b, a, size); __asm__ volatile("" ::: "memory"); }
    snprintf(name, sizeof name, "memcpy %s", tag); report(name, (double)size * reps / (now() - t) / 1e6, "MB/s");
    t = now(); for (int i = 0; i < reps; i++) { memset(b, i, size); __asm__ volatile("" ::: "memory"); }
    snprintf(name, sizeof name, "memset %s", tag); report(name, (double)size * reps / (now() - t) / 1e6, "MB/s");
    t = now(); for (int i = 0; i < reps; i++) rep_movsb(b, a, size);
    snprintf(name, sizeof name, "rep movsb %s", tag); report(name, (double)size * reps / (now() - t) / 1e6, "MB/s");
    t = now(); for (int i = 0; i < reps; i++) rep_stosb(b, i, size);
    snprintf(name, sizeof name, "rep stosb %s", tag); report(name, (double)size * reps / (now() - t) / 1e6, "MB/s");
    free(a); free(b);
}

static void bench_disk(const char *path, size_t total_mb) {
    size_t bs = 1 << 20; char *buf = aligned_alloc(4096, bs); memset(buf, 0x5a, bs);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { perror("open"); return; }
    double t = now();
    for (size_t i = 0; i < total_mb; i++) if (write(fd, buf, bs) != (ssize_t)bs) { perror("write"); return; }
    fsync(fd);
    report("disk seq write 1M", total_mb / (now() - t), "MB/s");
    lseek(fd, 0, SEEK_SET); t = now();
    for (size_t i = 0; i < total_mb; i++) if (read(fd, buf, bs) != (ssize_t)bs) { perror("read"); return; }
    report("disk seq read 1M", total_mb / (now() - t), "MB/s");
    long blocks = total_mb * 256, n = 20000; unsigned s = 12345; t = now();
    for (long i = 0; i < n; i++) { s = s * 1103515245 + 12345; if (pread(fd, buf, 4096, (off_t)(s % blocks) * 4096) != 4096) { perror("pread"); return; } }
    report("disk rand read 4K", n / (now() - t), "IOPS");
    t = now();
    for (long i = 0; i < n / 4; i++) { s = s * 1103515245 + 12345; if (pwrite(fd, buf, 4096, (off_t)(s % blocks) * 4096) != 4096) { perror("pwrite"); return; } }
    report("disk rand write 4K", n / 4 / (now() - t), "IOPS");
    close(fd); unlink(path);
}

static int dial(const char *ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0); struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port) };
    inet_pton(AF_INET, ip, &a.sin_addr);
    if (connect(s, (void *)&a, sizeof a) < 0) { perror("connect"); return -1; }
    return s;
}
static void bench_net(const char *ip, size_t total_mb) {
    size_t bs = 256 << 10; char *buf = malloc(bs); memset(buf, 7, bs);
    int s = dial(ip, 5201); if (s < 0) return; /* host sends total_mb MiB then closes */
    char req[32]; snprintf(req, sizeof req, "%zu\n", total_mb); write(s, req, strlen(req));
    double t = now(); size_t got = 0; ssize_t r;
    while ((r = read(s, buf, bs)) > 0) got += r;
    report("net rx (host->guest)", got / (now() - t) / 1e6, "MB/s"); close(s);
    s = dial(ip, 5202); if (s < 0) return; /* host sinks, replies "ok" at EOF */
    t = now();
    for (size_t sent = 0; sent < total_mb << 20; ) { r = write(s, buf, bs); if (r <= 0) break; sent += r; }
    shutdown(s, SHUT_WR); read(s, buf, 16);
    report("net tx (guest->host)", (double)(total_mb << 20) / (now() - t) / 1e6, "MB/s"); close(s);
    long n = 2000; s = dial(ip, 5203); int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    t = now();
    for (long i = 0; i < n; i++) { write(s, buf, 1); if (read(s, buf, 1) != 1) break; }
    report("net tcp rr 1B", n / (now() - t), "trans/s"); close(s);
}

int main(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    double w = now();
    if (!strcmp(what, "all") || !strcmp(what, "sys")) bench_syscall(2000000);
    if (!strcmp(what, "all") || !strcmp(what, "mem")) { bench_mem(4096, 200000, "4K"); bench_mem(64 << 10, 20000, "64K"); bench_mem(64 << 20, 10, "64M"); }
    if (!strcmp(what, "all") || !strcmp(what, "disk")) bench_disk(argc > 2 ? argv[2] : "/root/bench.dat", 512);
    if (!strcmp(what, "randread")) {
        const char *path = argc > 2 ? argv[2] : "/root/bench.dat";
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644); char *buf = aligned_alloc(4096, 1 << 20); memset(buf, 1, 1 << 20);
        for (int i = 0; i < 256; i++) write(fd, buf, 1 << 20);
        fsync(fd); long n = 200000; unsigned s = 1; double t = now();
        for (long i = 0; i < n; i++) { s = s * 1103515245 + 12345; pread(fd, buf, 4096, (off_t)(s % 65536) * 4096); }
        report("disk rand read 4K (long)", n / (now() - t), "IOPS"); close(fd); unlink(path);
    }
    if (!strcmp(what, "all") || !strcmp(what, "net")) bench_net(argc > 2 ? argv[2] : "10.0.2.2", 256);
    report("guest elapsed", now() - w, "s");
}
