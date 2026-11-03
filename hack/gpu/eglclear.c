/* Clears the screen through Mesa's virgl driver: EGL on a GBM surface on /dev/dri/card0, glClear,
 * and a KMS modeset of the result, which is a SET_SCANOUT and RESOURCE_FLUSH of a 3D resource.
 * The check for M0 of docs/plans/25-gpu-3d.md, built and run by hack/gpu/image.py.
 *
 *   eglclear [SECONDS]   red, green, blue, then red again, each held SECONDS (default 3); then
 *                        prints "EGLCLEAR done" and keeps the last frame until killed
 *
 * Prints the EGL and GL strings first ("GL_RENDERER: virgl (...)" is what says Mesa's virgl
 * driver is drawing and not llvmpipe), and the desktop GL version a surfaceless context gets.
 * Red, green and blue in that order show a swapped channel at a glance. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define CARD "/dev/dri/card0"

static void fail(const char *what) {
    fprintf(stderr, "EGLCLEAR failed: %s (EGL error 0x%x)\n", what, eglGetError());
    exit(1);
}

static void print_gl(const char *api) {
    printf("%s GL_VENDOR: %s\n", api, (const char *)glGetString(GL_VENDOR));
    printf("%s GL_RENDERER: %s\n", api, (const char *)glGetString(GL_RENDERER));
    printf("%s GL_VERSION: %s\n", api, (const char *)glGetString(GL_VERSION));
    printf("%s GL_SHADING_LANGUAGE_VERSION: %s\n", api,
           (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION));
}

/* Desktop GL on a surfaceless context, for the version Mesa computes from the capset. */
static void print_desktop_gl(EGLDisplay display) {
    if (!eglBindAPI(EGL_OPENGL_API)) {
        printf("desktop GL: no EGL_OPENGL_API\n");
        return;
    }
    static const EGLint config_attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
    EGLConfig config;
    EGLint count = 0;
    if (!eglChooseConfig(display, config_attributes, &config, 1, &count) || count == 0) {
        printf("desktop GL: no config\n");
        return;
    }
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    if (context == EGL_NO_CONTEXT ||
        !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
        printf("desktop GL: no surfaceless context (0x%x)\n", eglGetError());
        return;
    }
    print_gl("desktop");
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
}

int main(int argc, char **argv) {
    int hold = argc > 1 ? atoi(argv[1]) : 3;
    setvbuf(stdout, NULL, _IONBF, 0);

    int fd = open(CARD, O_RDWR | O_CLOEXEC);
    if (fd < 0) fail("open " CARD);
    drmModeRes *resources = drmModeGetResources(fd);
    if (!resources) fail("drmModeGetResources");
    drmModeConnector *connector = NULL;
    for (int i = 0; i < resources->count_connectors && !connector; i++) {
        drmModeConnector *candidate = drmModeGetConnector(fd, resources->connectors[i]);
        if (candidate && candidate->connection == DRM_MODE_CONNECTED && candidate->count_modes > 0)
            connector = candidate;
    }
    if (!connector) fail("no connected connector");
    drmModeModeInfo mode = connector->modes[0];
    uint32_t crtc = resources->crtcs[0];
    printf("mode %ux%u on connector %u, crtc %u\n", mode.hdisplay, mode.vdisplay,
           connector->connector_id, crtc);

    struct gbm_device *gbm = gbm_create_device(fd);
    if (!gbm) fail("gbm_create_device");
    struct gbm_surface *surface = gbm_surface_create(gbm, mode.hdisplay, mode.vdisplay,
                                                     GBM_FORMAT_XRGB8888,
                                                     GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!surface) fail("gbm_surface_create");

    EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    EGLint major, minor;
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) fail("eglInitialize");
    printf("EGL_VENDOR: %s\nEGL_VERSION: %s\n", eglQueryString(display, EGL_VENDOR),
           eglQueryString(display, EGL_VERSION));
    print_desktop_gl(display);

    if (!eglBindAPI(EGL_OPENGL_ES_API)) fail("eglBindAPI");
    static const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE,
    };
    EGLConfig configs[64];
    EGLint count = 0;
    eglChooseConfig(display, config_attributes, configs, 64, &count);
    /* The config must match the GBM surface's format, or eglCreateWindowSurface refuses it. */
    EGLConfig config = NULL;
    for (int i = 0; i < count && !config; i++) {
        EGLint visual;
        if (eglGetConfigAttrib(display, configs[i], EGL_NATIVE_VISUAL_ID, &visual) &&
            visual == GBM_FORMAT_XRGB8888)
            config = configs[i];
    }
    if (!config) fail("no XRGB8888 config");
    static const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
    if (context == EGL_NO_CONTEXT) fail("eglCreateContext");
    EGLSurface window = eglCreateWindowSurface(display, config, (EGLNativeWindowType)surface, NULL);
    if (window == EGL_NO_SURFACE) fail("eglCreateWindowSurface");
    if (!eglMakeCurrent(display, window, window, context)) fail("eglMakeCurrent");
    print_gl("GLES");

    static const struct { const char *name; float rgb[3]; } colors[] = {
        {"red", {1, 0, 0}}, {"green", {0, 1, 0}}, {"blue", {0, 0, 1}}, {"red", {1, 0, 0}},
    };
    struct gbm_bo *shown = NULL;
    uint32_t shown_fb = 0;
    for (unsigned i = 0; i < sizeof colors / sizeof colors[0]; i++) {
        glClearColor(colors[i].rgb[0], colors[i].rgb[1], colors[i].rgb[2], 1);
        glClear(GL_COLOR_BUFFER_BIT);
        if (!eglSwapBuffers(display, window)) fail("eglSwapBuffers");
        struct gbm_bo *bo = gbm_surface_lock_front_buffer(surface);
        if (!bo) fail("gbm_surface_lock_front_buffer");
        uint32_t handles[4] = {gbm_bo_get_handle(bo).u32};
        uint32_t pitches[4] = {gbm_bo_get_stride(bo)};
        uint32_t offsets[4] = {0};
        uint32_t fb;
        if (drmModeAddFB2(fd, mode.hdisplay, mode.vdisplay, GBM_FORMAT_XRGB8888, handles, pitches,
                          offsets, &fb, 0))
            fail("drmModeAddFB2");
        if (drmModeSetCrtc(fd, crtc, fb, 0, 0, &connector->connector_id, 1, &mode))
            fail("drmModeSetCrtc");
        if (shown) {
            drmModeRmFB(fd, shown_fb);
            gbm_surface_release_buffer(surface, shown);
        }
        shown = bo;
        shown_fb = fb;
        printf("EGLCLEAR frame %s\n", colors[i].name);
        sleep(hold);
    }
    printf("EGLCLEAR done\n");
    pause();
    return 0;
}
