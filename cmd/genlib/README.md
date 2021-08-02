# genlib #

genlib is the program that generates the package `cu`. It does so by parsing `cuda.h` which is a modified form of `cuda.h` that comes with a CUDA installation. Specifially these were the modifications made:

```
cp /usr/local/cuda/include/cuda.h cuda2.h // copy cuda.h to current dir
echo 'include "cuda2.h"' > cuda.c         // make a fake C file
gcc -E -P cuda.c > cuda.h                 // ask GCC to perform preprocessing.
astyle --style=google --lineend=linux --indent=tab --indent-switches --align-pointer=type --align-reference=name --delete-empty-lines cuda.h // fmt this
sed -i 's/_v2//g' cuda.h                 // Remove _v2 stufff from cuda.h
sed -i 's/_v3//g' cuda.h                 // Remove _v3 stuff from cuda.h
sed -i -E 's/^#.+//g' cuda.h
sed -i '/^$/N;/^\n$/D' cuda.h
```


The first line preprocesses all the macros, leaving a singular header file. The second command processes the files in a way that is readable to me (and the generator program). THe last two commands replaces any v2/v3 that may be found The copyright notice from nVidia is then reattached.

After that, the file is manually fixed by means of running `go run *.go`. The errors/panics are all related to parsing of C files (e.g. `unnamed fields not allowed`). These are manually fixed one by one thus:

* `unnamed fields not allowed` - give said fields dummy names

When the program finds an error, it will leave the file ungenerated, reporting errors instead. This allows for new versions of `cuda.h` to be adapted quickly.

Here's an example moving from CUDA 9.0 to CUDA 11.0's API (CUDA10 didn't require all these changes so I postponed making them).

When upgrading to support CUDA11, there were many new constructs that were introduced that needed a manual Go translation. The errors are that the `ctype`s are not known to the translator. The following line will simply output them.

```
$go run *.go 2>&1 >/dev/null | grep ctype > TODO
```

Once this list is gotten, we can then either

a) manually make corresponding Go data structures.
b) generate enums (if they are enums).
c) ignore them.


# Manually fixed after generation #

* `MemsetD32`
* `CurrentFlags`
* `SetBorderColor`
* `BorderColor`
* `PrimaryCtxState`

## Ctx related methods ##
* `(*Ctx)SetPrimaryCtxFlags`
* `SynchronizeStream`
* `SynchronizeEvent`
* `APIVersion` - deleted
* `QueryStream`
* `QueryEvent`
* `FunctionAttribute`
* `DeviceAttribute`
* `SetFunctionSharedMemConfig`
* `SetTexRefFlags`
* `TexRefFlags`
* `StreamFlags`
* `TexRefSetArray`
* `SurfRefSetArray`
* `SetBorderColor`
* `BorderColor`
* `NumDevices` - deleted
* `TotalMem` - deleted
* `DeviceAttribute` - deleted
* `GetDevice` - deleted
* `ReleasePrimaryCtx` - deleted
* `SetPrimaryCtxFlags` - deleted
* `PrimaryCtxState` - deleted
* `ResetPrimaryCtx` - deleted
* `PushCurrentCtx` - deleted
* `PopCurrentCtx` - deleted
* `SetCurrentContext` - deleted
* `CurrentContext` - deleted
* `CurrentDevice`
* `CurrentFlags`
* `CanAccessPeer` - deleted
* `P2PAttribute` - deleted
* `MemAllocManaged`

## Ctx related methods - manually written ##
