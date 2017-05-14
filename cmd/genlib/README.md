# genlibcu #

genlibcu is the program that generates the package `cu`. It does so by parsing `cuda.h` which is a modified form of `cuda.h` that comes with a CUDA installation. Specifially these were the modifications made:

```
gcc -E cuda.h > cuda.h
astyle --style=google --lineend=linux --indent=tab --indent-switches --align-pointer=type --align-reference=name --delete-empty-lines cuda.h
sed -i 's/_v2//g' cuda.h
sed -i 's/_v3//g' cuda.h
```

The first line preprocesses all the macros, leaving a singular header file. The second command processes the files in a way that is readable to me (and the generator program). THe last two commands replaces any v2/v3 that may be found The copyright notice from nVidia is then reattached.

# Manually fixed after generation #

* `MemsetD32`
* `CurrentFlags`
* `SetBorderColor`
* `BorderColor`

## Ctx related methods ##
* `(*Ctx)SetPrimaryCtxFlags`
* `PushCurrentCtx`
* `SetCurrentContext`
* `SynchronizeStream`
* `SynchronizeEvent`
* `APIVersion` - deleted
* `QueryStream`
* `QueryEvent`
* `FunctionAttribute`
* `DeviceAttribute`
* `SetFunctionSharedMemConfig` 
* `TexRefFlags`
* `StreamFlags`
* `TexRefSetArray`
* `SurfRefSetArray`
* `SetBorderColor`
* `BorderColor`