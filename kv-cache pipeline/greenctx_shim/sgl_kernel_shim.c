#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>

// Shim for torch ABI mismatch: provide missing symbols expected by sgl_kernel.

typedef void (*set_device_fn_t)(signed char);

void _ZN3c104cuda9SetDeviceEab(signed char device, bool _maybe_unused) {
    (void)_maybe_unused;
    static set_device_fn_t fn = NULL;
    if (!fn) {
        fn = (set_device_fn_t)dlsym(RTLD_DEFAULT, "_ZN3c104cuda9SetDeviceEa");
    }
    if (fn) {
        fn(device);
    }
}

// c10::SymBool::guard_or_false(char const*, long) const
bool _ZNK3c107SymBool14guard_or_falseEPKcl(const void* _self, const char* _msg, long _line) {
    (void)_self;
    (void)_msg;
    (void)_line;
    return false;
}
