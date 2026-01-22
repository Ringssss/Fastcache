#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>

#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <optional>

using c10::SymBool;
using c10::SymInt;

// Shim for torch ABI mismatch: provide missing symbols expected by sgl_kernel.

typedef void (*set_device_fn_t)(signed char);

extern "C" void _ZN3c104cuda9SetDeviceEab(signed char device, bool _maybe_unused) {
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
extern "C" bool _ZNK3c107SymBool14guard_or_falseEPKcl(
    const SymBool* _self,
    const char* _file,
    long _line) {
    (void)_self;
    (void)_file;
    (void)_line;
    return false;
}

// c10::SymInt::maybe_as_int_slow_path() const
extern "C" std::optional<int64_t> _ZNK3c106SymInt22maybe_as_int_slow_pathEv(
    const SymInt* _self) {
    (void)_self;
    return std::nullopt;
}

// c10::SymInt::sym_ne_slow_path(c10::SymInt const&) const
extern "C" SymBool _ZNK3c106SymInt16sym_ne_slow_pathERKS0_(
    const SymInt* _self,
    const SymInt* _other) {
    (void)_self;
    (void)_other;
    return SymBool(false);
}
