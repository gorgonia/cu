# genlibcu #

genlibcu is the program that generates the package `cu`. It does so by parsing `cuda.h` which is a modified form of `cuda.h` that comes with a CUDA installation. Specifially these were the modifications made:

```
gcc -E -P cuda.h > cuda.h
astyle --style=google --lineend=linux --indent=tab --indent-switches --align-pointer=type --align-reference=name --delete-empty-lines cuda.h
sed -i 's/_v2//g' cuda.h
sed -i 's/_v3//g' cuda.h
sed -i -E 's/^#.+//g' cuda.h
sed -i '/^$/N;/^\n$/D' cuda.h
```

The first line preprocesses all the macros, leaving a singular header file. The second command processes the files in a way that is readable to me (and the generator program). THe last two commands replaces any v2/v3 that may be found The copyright notice from nVidia is then reattached.

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