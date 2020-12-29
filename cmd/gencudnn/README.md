
1. Preprocess the cudnn file: `gcc -E -P cudnn_.h > cudnn.h`
2. Add the following "headers" to cheat the compiler:
	```
		typedef long int ptrdiff_t;
		typedef long unsigned int size_t;
		typedef long unsigned int rsize_t;
		typedef int wchar_t;
		typedef long double max_align_t;
		struct dummy;
		typedef struct dummy cudaStream_t;
	```
3. Delete the following debug related stuff (only cuDNN 7.1+):
	```
	typedef struct {
	    unsigned cudnn_version;
	    cudnnStatus_t cudnnStatus;
	    unsigned time_sec;
	    unsigned time_usec;
	    unsigned time_delta;
	    cudnnHandle_t handle;
	    cudaStream_t stream;
	    unsigned long long pid;
	    unsigned long long tid;
	    int cudaDeviceId;
	    int reserved[15];
	} cudnnDebug_t;
	typedef void (*cudnnCallback_t) (cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);
cudnnStatus_t cudnnSetCallback(
                                unsigned mask,
                                void *udata,
                                cudnnCallback_t fptr);
cudnnStatus_t cudnnGetCallback(
                                unsigned *mask,
                                void **udata,
                                cudnnCallback_t *fptr);
	```


# TODOs

## Stubs ##

*  /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_ctcloss.go. TODO: true
*  ~~/home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_spatialtransformer.go. TODO: true~~
*  ~~/home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_seqdata.go. TODO: true~~
*  ~~/home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_backend.go. TODO: true~~
*  ~~/home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_rnndata.go. TODO: true~~
*  ~~/home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_tensortransform.go. TODO: true~~
*  /home/chewxy/workspace/gorgoniaws/src/gorgonia.org/cu/dnn/generated_algorithmdescriptor.go. TODO: true
