typedef long unsigned int size_t;
typedef int wchar_t;

typedef enum {
	P_ALL,
	P_PID,
	P_PGID
} idtype_t;
typedef struct {
	int quot;
	int rem;
} div_t;
typedef struct {
	long int quot;
	long int rem;
} ldiv_t;
__extension__ typedef struct {
	long long int quot;
	long long int rem;
} lldiv_t;
extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__, __leaf__)) ;
extern double atof (const char* __nptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;
extern int atoi (const char* __nptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;
extern long int atol (const char* __nptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;
__extension__ extern long long int atoll (const char* __nptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;
extern double strtod (const char* __restrict __nptr,
                      char** __restrict __endptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern float strtof (const char* __restrict __nptr,
                     char** __restrict __endptr) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern long double strtold (const char* __restrict __nptr,
                            char** __restrict __endptr)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern long int strtol (const char* __restrict __nptr,
                        char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern unsigned long int strtoul (const char* __restrict __nptr,
                                  char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
__extension__
extern long long int strtoq (const char* __restrict __nptr,
                             char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
__extension__
extern unsigned long long int strtouq (const char* __restrict __nptr,
                                       char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
__extension__
extern long long int strtoll (const char* __restrict __nptr,
                              char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
__extension__
extern unsigned long long int strtoull (const char* __restrict __nptr,
                                        char** __restrict __endptr, int __base)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern char* l64a (long int __n) __attribute__ ((__nothrow__, __leaf__)) ;
extern long int a64l (const char* __s)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) ;

typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef long int __intmax_t;
typedef unsigned long int __uintmax_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct {
	int __val[2];
} __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef int __daddr_t;
typedef int __key_t;
typedef int __clockid_t;
typedef void* __timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __fsword_t;
typedef long int __ssize_t;
typedef long int __syscall_slong_t;
typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef char* __caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef int __sig_atomic_t;
typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;
typedef __loff_t loff_t;
typedef __ino_t ino_t;
typedef __dev_t dev_t;
typedef __gid_t gid_t;
typedef __mode_t mode_t;
typedef __nlink_t nlink_t;
typedef __uid_t uid_t;
typedef __off_t off_t;
typedef __pid_t pid_t;
typedef __id_t id_t;
typedef __ssize_t ssize_t;
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;
typedef __key_t key_t;
typedef __clock_t clock_t;
typedef __clockid_t clockid_t;
typedef __time_t time_t;
typedef __timer_t timer_t;
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef __int8_t int8_t;
typedef __int16_t int16_t;
typedef __int32_t int32_t;
typedef __int64_t int64_t;
typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));
typedef int register_t __attribute__ ((__mode__ (__word__)));
static __inline unsigned int
__bswap_32 (unsigned int __bsx) {
	return __builtin_bswap32 (__bsx);
}
static __inline __uint64_t
__bswap_64 (__uint64_t __bsx) {
	return __builtin_bswap64 (__bsx);
}
static __inline __uint16_t
__uint16_identity (__uint16_t __x) {
	return __x;
}
static __inline __uint32_t
__uint32_identity (__uint32_t __x) {
	return __x;
}
static __inline __uint64_t
__uint64_identity (__uint64_t __x) {
	return __x;
}
typedef struct {
	unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
} __sigset_t;
typedef __sigset_t sigset_t;
struct timeval {
	__time_t tv_sec;
	__suseconds_t tv_usec;
};
struct timespec {
	__time_t tv_sec;
	__syscall_slong_t tv_nsec;
};
typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
typedef struct {
	__fd_mask __fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];
} fd_set;
typedef __fd_mask fd_mask;

extern int select (int __nfds, fd_set* __restrict __readfds,
                   fd_set* __restrict __writefds,
                   fd_set* __restrict __exceptfds,
                   struct timeval* __restrict __timeout);
extern int pselect (int __nfds, fd_set* __restrict __readfds,
                    fd_set* __restrict __writefds,
                    fd_set* __restrict __exceptfds,
                    const struct timespec* __restrict __timeout,
                    const __sigset_t* __restrict __sigmask);

extern unsigned int gnu_dev_major (__dev_t __dev) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__));
extern unsigned int gnu_dev_minor (__dev_t __dev) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__));
extern __dev_t gnu_dev_makedev (unsigned int __major, unsigned int __minor) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__));

typedef __blksize_t blksize_t;
typedef __blkcnt_t blkcnt_t;
typedef __fsblkcnt_t fsblkcnt_t;
typedef __fsfilcnt_t fsfilcnt_t;
struct __pthread_rwlock_arch_t {
	unsigned int __readers;
	unsigned int __writers;
	unsigned int __wrphase_futex;
	unsigned int __writers_futex;
	unsigned int __pad3;
	unsigned int __pad4;
	int __cur_writer;
	int __shared;
	signed char __rwelision;
	unsigned char __pad1[7];
	unsigned long int __pad2;
	unsigned int __flags;
};
typedef struct __pthread_internal_list {
	struct __pthread_internal_list* __prev;
	struct __pthread_internal_list* __next;
} __pthread_list_t;
struct __pthread_mutex_s {
	int __lock ;
	unsigned int __count;
	int __owner;
	unsigned int __nusers;
	int __kind;

	short __spins;
	short __elision;
	__pthread_list_t __list;

};
struct __pthread_cond_s {
	__extension__ union {
		__extension__ unsigned long long int __wseq;
		struct {
			unsigned int __low;
			unsigned int __high;
		} __wseq32;
	} foo;
	__extension__ union {
		__extension__ unsigned long long int __g1_start;
		struct {
			unsigned int __low;
			unsigned int __high;
		} __g1_start32;
	} bar ;
	unsigned int __g_refs[2] ;
	unsigned int __g_size[2];
	unsigned int __g1_orig_size;
	unsigned int __wrefs;
	unsigned int __g_signals[2];
};
typedef unsigned long int pthread_t;
typedef union {
	char __size[4];
	int __align;
} pthread_mutexattr_t;
typedef union {
	char __size[4];
	int __align;
} pthread_condattr_t;
typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
union pthread_attr_t {
	char __size[56];
	long int __align;
};
typedef union pthread_attr_t pthread_attr_t;
typedef union {
	struct __pthread_mutex_s __data;
	char __size[40];
	long int __align;
} pthread_mutex_t;
typedef union {
	struct __pthread_cond_s __data;
	char __size[48];
	__extension__ long long int __align;
} pthread_cond_t;
typedef union {
	struct __pthread_rwlock_arch_t __data;
	char __size[56];
	long int __align;
} pthread_rwlock_t;
typedef union {
	char __size[8];
	long int __align;
} pthread_rwlockattr_t;
typedef volatile int pthread_spinlock_t;
typedef union {
	char __size[32];
	long int __align;
} pthread_barrier_t;
typedef union {
	char __size[4];
	int __align;
} pthread_barrierattr_t;

extern long int random (void) __attribute__ ((__nothrow__, __leaf__));
extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__, __leaf__));
extern char* initstate (unsigned int __seed, char* __statebuf,
                        size_t __statelen) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (2)));
extern char* setstate (char* __statebuf) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
struct random_data {
	int32_t* fptr;
	int32_t* rptr;
	int32_t* state;
	int rand_type;
	int rand_deg;
	int rand_sep;
	int32_t* end_ptr;
};
extern int random_r (struct random_data* __restrict __buf,
                     int32_t* __restrict __result) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int srandom_r (unsigned int __seed, struct random_data* __buf)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (2)));
extern int initstate_r (unsigned int __seed, char* __restrict __statebuf,
                        size_t __statelen,
                        struct random_data* __restrict __buf)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (2, 4)));
extern int setstate_r (char* __restrict __statebuf,
                       struct random_data* __restrict __buf)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int rand (void) __attribute__ ((__nothrow__, __leaf__));
extern void srand (unsigned int __seed) __attribute__ ((__nothrow__, __leaf__));
extern int rand_r (unsigned int* __seed) __attribute__ ((__nothrow__, __leaf__));
extern double drand48 (void) __attribute__ ((__nothrow__, __leaf__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern long int lrand48 (void) __attribute__ ((__nothrow__, __leaf__));
extern long int nrand48 (unsigned short int __xsubi[3])
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern long int mrand48 (void) __attribute__ ((__nothrow__, __leaf__));
extern long int jrand48 (unsigned short int __xsubi[3])
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void srand48 (long int __seedval) __attribute__ ((__nothrow__, __leaf__));
extern unsigned short int* seed48 (unsigned short int __seed16v[3])
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
struct drand48_data {
	unsigned short int __x[3];
	unsigned short int __old_x[3];
	unsigned short int __c;
	unsigned short int __init;
	__extension__ unsigned long long int __a;
};
extern int drand48_r (struct drand48_data* __restrict __buffer,
                      double* __restrict __result) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
                      struct drand48_data* __restrict __buffer,
                      double* __restrict __result) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int lrand48_r (struct drand48_data* __restrict __buffer,
                      long int* __restrict __result)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
                      struct drand48_data* __restrict __buffer,
                      long int* __restrict __result)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int mrand48_r (struct drand48_data* __restrict __buffer,
                      long int* __restrict __result)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
                      struct drand48_data* __restrict __buffer,
                      long int* __restrict __result)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int srand48_r (long int __seedval, struct drand48_data* __buffer)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (2)));
extern int seed48_r (unsigned short int __seed16v[3],
                     struct drand48_data* __buffer) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int lcong48_r (unsigned short int __param[7],
                      struct drand48_data* __buffer)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern void* malloc (size_t __size) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__malloc__)) ;
extern void* calloc (size_t __nmemb, size_t __size)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__malloc__)) ;
extern void* realloc (void* __ptr, size_t __size)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__warn_unused_result__));
extern void free (void* __ptr) __attribute__ ((__nothrow__, __leaf__));

extern void* alloca (size_t __size) __attribute__ ((__nothrow__, __leaf__));

extern void* valloc (size_t __size) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__malloc__)) ;
extern int posix_memalign (void** __memptr, size_t __alignment, size_t __size)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1))) ;
extern void* aligned_alloc (size_t __alignment, size_t __size)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__alloc_size__ (2))) ;
extern void abort (void) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int at_quick_exit (void (*__func) (void)) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int on_exit (void (*__func) (int __status, void* __arg), void* __arg)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void exit (int __status) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
extern void quick_exit (int __status) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
extern void _Exit (int __status) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
extern char* getenv (const char* __name) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1))) ;
extern int putenv (char* __string) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int setenv (const char* __name, const char* __value, int __replace)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (2)));
extern int unsetenv (const char* __name) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int clearenv (void) __attribute__ ((__nothrow__, __leaf__));
extern char* mktemp (char* __template) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));
extern int mkstemp (char* __template) __attribute__ ((__nonnull__ (1))) ;
extern int mkstemps (char* __template, int __suffixlen) __attribute__ ((__nonnull__ (1))) ;
extern char* mkdtemp (char* __template) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1))) ;
extern int system (const char* __command) ;
extern char* realpath (const char* __restrict __name,
                       char* __restrict __resolved) __attribute__ ((__nothrow__, __leaf__)) ;
typedef int (*__compar_fn_t) (const void*, const void*);
extern void* bsearch (const void* __key, const void* __base,
                      size_t __nmemb, size_t __size, __compar_fn_t __compar)
__attribute__ ((__nonnull__ (1, 2, 5))) ;
extern void qsort (void* __base, size_t __nmemb, size_t __size,
                   __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
extern int abs (int __x) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
extern long int labs (long int __x) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
__extension__ extern long long int llabs (long long int __x)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
extern div_t div (int __numer, int __denom)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
extern ldiv_t ldiv (long int __numer, long int __denom)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
__extension__ extern lldiv_t lldiv (long long int __numer,
                                    long long int __denom)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__const__)) ;
extern char* ecvt (double __value, int __ndigit, int* __restrict __decpt,
                   int* __restrict __sign) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char* fcvt (double __value, int __ndigit, int* __restrict __decpt,
                   int* __restrict __sign) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char* gcvt (double __value, int __ndigit, char* __buf)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3))) ;
extern char* qecvt (long double __value, int __ndigit,
                    int* __restrict __decpt, int* __restrict __sign)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char* qfcvt (long double __value, int __ndigit,
                    int* __restrict __decpt, int* __restrict __sign)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4))) ;
extern char* qgcvt (long double __value, int __ndigit, char* __buf)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3))) ;
extern int ecvt_r (double __value, int __ndigit, int* __restrict __decpt,
                   int* __restrict __sign, char* __restrict __buf,
                   size_t __len) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int* __restrict __decpt,
                   int* __restrict __sign, char* __restrict __buf,
                   size_t __len) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qecvt_r (long double __value, int __ndigit,
                    int* __restrict __decpt, int* __restrict __sign,
                    char* __restrict __buf, size_t __len)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
                    int* __restrict __decpt, int* __restrict __sign,
                    char* __restrict __buf, size_t __len)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int mblen (const char* __s, size_t __n) __attribute__ ((__nothrow__, __leaf__));
extern int mbtowc (wchar_t* __restrict __pwc,
                   const char* __restrict __s, size_t __n) __attribute__ ((__nothrow__, __leaf__));
extern int wctomb (char* __s, wchar_t __wchar) __attribute__ ((__nothrow__, __leaf__));
extern size_t mbstowcs (wchar_t* __restrict __pwcs,
                        const char* __restrict __s, size_t __n) __attribute__ ((__nothrow__, __leaf__));
extern size_t wcstombs (char* __restrict __s,
                        const wchar_t* __restrict __pwcs, size_t __n)
__attribute__ ((__nothrow__, __leaf__));
extern int rpmatch (const char* __response) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1))) ;
extern int getsubopt (char** __restrict __optionp,
                      char* const* __restrict __tokens,
                      char** __restrict __valuep)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1, 2, 3))) ;
extern int getloadavg (double __loadavg[], int __nelem)
__attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__nonnull__ (1)));

typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;
typedef long int int_least64_t;
typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long int uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef long int intptr_t;
typedef unsigned long int uintptr_t;
typedef __intmax_t intmax_t;
typedef __uintmax_t uintmax_t;
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;
typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUarray_st* CUarray;
typedef struct CUmipmappedArray_st* CUmipmappedArray;
typedef struct CUtexref_st* CUtexref;
typedef struct CUsurfref_st* CUsurfref;
typedef struct CUevent_st* CUevent;
typedef struct CUstream_st* CUstream;
typedef struct CUgraphicsResource_st* CUgraphicsResource;
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;
typedef struct CUextMemory_st* CUexternalMemory;
typedef struct CUextSemaphore_st* CUexternalSemaphore;
typedef struct CUgraph_st* CUgraph;
typedef struct CUgraphNode_st* CUgraphNode;
typedef struct CUgraphExec_st* CUgraphExec;
typedef struct CUuuid_st {
	char bytes[16];
} CUuuid;
typedef struct CUipcEventHandle_st {
	char reserved[64];
} CUipcEventHandle;
typedef struct CUipcMemHandle_st {
	char reserved[64];
} CUipcMemHandle;
typedef enum CUipcMem_flags_enum {
	CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
} CUipcMem_flags;
typedef enum CUmemAttach_flags_enum {
	CU_MEM_ATTACH_GLOBAL = 0x1,
	CU_MEM_ATTACH_HOST = 0x2,
	CU_MEM_ATTACH_SINGLE = 0x4
} CUmemAttach_flags;
typedef enum CUctx_flags_enum {
	CU_CTX_SCHED_AUTO = 0x00,
	CU_CTX_SCHED_SPIN = 0x01,
	CU_CTX_SCHED_YIELD = 0x02,
	CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
	CU_CTX_BLOCKING_SYNC = 0x04,
	CU_CTX_SCHED_MASK = 0x07,
	CU_CTX_MAP_HOST = 0x08,
	CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
	CU_CTX_FLAGS_MASK = 0x1f
} CUctx_flags;
typedef enum CUstream_flags_enum {
	CU_STREAM_DEFAULT = 0x0,
	CU_STREAM_NON_BLOCKING = 0x1
} CUstream_flags;
typedef enum CUevent_flags_enum {
	CU_EVENT_DEFAULT = 0x0,
	CU_EVENT_BLOCKING_SYNC = 0x1,
	CU_EVENT_DISABLE_TIMING = 0x2,
	CU_EVENT_INTERPROCESS = 0x4
} CUevent_flags;
typedef enum CUstreamWaitValue_flags_enum {
	CU_STREAM_WAIT_VALUE_GEQ = 0x0,
	CU_STREAM_WAIT_VALUE_EQ = 0x1,
	CU_STREAM_WAIT_VALUE_AND = 0x2,
	CU_STREAM_WAIT_VALUE_NOR = 0x3,
	CU_STREAM_WAIT_VALUE_FLUSH = 1<<30
} CUstreamWaitValue_flags;
typedef enum CUstreamWriteValue_flags_enum {
	CU_STREAM_WRITE_VALUE_DEFAULT = 0x0,
	CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1
} CUstreamWriteValue_flags;
typedef enum CUstreamBatchMemOpType_enum {
	CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
	CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
	CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
	CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
	CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
} CUstreamBatchMemOpType;
typedef union CUstreamBatchMemOpParams_union {
	CUstreamBatchMemOpType operation;
	struct CUstreamMemOpWaitValueParams_st {
		CUstreamBatchMemOpType operation;
		CUdeviceptr address;
		union {
			cuuint32_t value;
			cuuint64_t value64;
		} foo;
		unsigned int flags;
		CUdeviceptr alias;
	} waitValue;
	struct CUstreamMemOpWriteValueParams_st {
		CUstreamBatchMemOpType operation;
		CUdeviceptr address;
		union {
			cuuint32_t value;
			cuuint64_t value64;
		} bar;
		unsigned int flags;
		CUdeviceptr alias;
	} writeValue;
	struct CUstreamMemOpFlushRemoteWritesParams_st {
		CUstreamBatchMemOpType operation;
		unsigned int flags;
	} flushRemoteWrites;
	cuuint64_t pad[6];
} CUstreamBatchMemOpParams;
typedef enum CUoccupancy_flags_enum {
	CU_OCCUPANCY_DEFAULT = 0x0,
	CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1
} CUoccupancy_flags;
typedef enum CUarray_format_enum {
	CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
	CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
	CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
	CU_AD_FORMAT_SIGNED_INT8 = 0x08,
	CU_AD_FORMAT_SIGNED_INT16 = 0x09,
	CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
	CU_AD_FORMAT_HALF = 0x10,
	CU_AD_FORMAT_FLOAT = 0x20
} CUarray_format;
typedef enum CUaddress_mode_enum {
	CU_TR_ADDRESS_MODE_WRAP = 0,
	CU_TR_ADDRESS_MODE_CLAMP = 1,
	CU_TR_ADDRESS_MODE_MIRROR = 2,
	CU_TR_ADDRESS_MODE_BORDER = 3
} CUaddress_mode;
typedef enum CUfilter_mode_enum {
	CU_TR_FILTER_MODE_POINT = 0,
	CU_TR_FILTER_MODE_LINEAR = 1
} CUfilter_mode;
typedef enum CUdevice_attribute_enum {
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
	CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
	CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
	CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
	CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
	CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
	CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
	CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
	CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
	CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
	CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
	CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
	CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
	CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
	CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
	CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
	CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
	CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
	CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
	CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
	CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
	CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
	CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
	CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
	CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
	CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
	CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
	CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
	CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
	CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
	CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
	CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
	CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
	CU_DEVICE_ATTRIBUTE_MAX
} CUdevice_attribute;
typedef struct CUdevprop_st {
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	int sharedMemPerBlock;
	int totalConstantMemory;
	int SIMDWidth;
	int memPitch;
	int regsPerBlock;
	int clockRate;
	int textureAlign;
} CUdevprop;
typedef enum CUpointer_attribute_enum {
	CU_POINTER_ATTRIBUTE_CONTEXT = 1,
	CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
	CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
	CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
	CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
	CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
	CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
	CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
	CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
	CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
	CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
	CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
	CU_POINTER_ATTRIBUTE_MAPPED = 13,
	CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
	CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
} CUpointer_attribute;
typedef enum CUfunction_attribute_enum {
	CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
	CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
	CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
	CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
	CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
	CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
	CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
	CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
	CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
	CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
	CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;
typedef enum CUfunc_cache_enum {
	CU_FUNC_CACHE_PREFER_NONE = 0x00,
	CU_FUNC_CACHE_PREFER_SHARED = 0x01,
	CU_FUNC_CACHE_PREFER_L1 = 0x02,
	CU_FUNC_CACHE_PREFER_EQUAL = 0x03
} CUfunc_cache;
typedef enum CUsharedconfig_enum {
	CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00,
	CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01,
	CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02
} CUsharedconfig;
typedef enum CUshared_carveout_enum {
	CU_SHAREDMEM_CARVEOUT_DEFAULT = -1,
	CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
	CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0
} CUshared_carveout;
typedef enum CUmemorytype_enum {
	CU_MEMORYTYPE_HOST = 0x01,
	CU_MEMORYTYPE_DEVICE = 0x02,
	CU_MEMORYTYPE_ARRAY = 0x03,
	CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;
typedef enum CUcomputemode_enum {
	CU_COMPUTEMODE_DEFAULT = 0,
	CU_COMPUTEMODE_PROHIBITED = 2,
	CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
} CUcomputemode;
typedef enum CUmem_advise_enum {
	CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
	CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
	CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
	CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
	CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
	CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
} CUmem_advise;
typedef enum CUmem_range_attribute_enum {
	CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
	CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
	CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
	CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
} CUmem_range_attribute;
typedef enum CUjit_option_enum {
	CU_JIT_MAX_REGISTERS = 0,
	CU_JIT_THREADS_PER_BLOCK,
	CU_JIT_WALL_TIME,
	CU_JIT_INFO_LOG_BUFFER,
	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
	CU_JIT_ERROR_LOG_BUFFER,
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
	CU_JIT_OPTIMIZATION_LEVEL,
	CU_JIT_TARGET_FROM_CUCONTEXT,
	CU_JIT_TARGET,
	CU_JIT_FALLBACK_STRATEGY,
	CU_JIT_GENERATE_DEBUG_INFO,
	CU_JIT_LOG_VERBOSE,
	CU_JIT_GENERATE_LINE_INFO,
	CU_JIT_CACHE_MODE,
	CU_JIT_NEW_SM3X_OPT,
	CU_JIT_FAST_COMPILE,
	CU_JIT_GLOBAL_SYMBOL_NAMES,
	CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
	CU_JIT_GLOBAL_SYMBOL_COUNT,
	CU_JIT_NUM_OPTIONS
} CUjit_option;
typedef enum CUjit_target_enum {
	CU_TARGET_COMPUTE_20 = 20,
	CU_TARGET_COMPUTE_21 = 21,
	CU_TARGET_COMPUTE_30 = 30,
	CU_TARGET_COMPUTE_32 = 32,
	CU_TARGET_COMPUTE_35 = 35,
	CU_TARGET_COMPUTE_37 = 37,
	CU_TARGET_COMPUTE_50 = 50,
	CU_TARGET_COMPUTE_52 = 52,
	CU_TARGET_COMPUTE_53 = 53,
	CU_TARGET_COMPUTE_60 = 60,
	CU_TARGET_COMPUTE_61 = 61,
	CU_TARGET_COMPUTE_62 = 62,
	CU_TARGET_COMPUTE_70 = 70,
	CU_TARGET_COMPUTE_72 = 72,
	CU_TARGET_COMPUTE_75 = 75,
	CU_TARGET_COMPUTE_80 = 80
} CUjit_target;
typedef enum CUjit_fallback_enum {
	CU_PREFER_PTX = 0,
	CU_PREFER_BINARY
} CUjit_fallback;
typedef enum CUjit_cacheMode_enum {
	CU_JIT_CACHE_OPTION_NONE = 0,
	CU_JIT_CACHE_OPTION_CG,
	CU_JIT_CACHE_OPTION_CA
} CUjit_cacheMode;
typedef enum CUjitInputType_enum {
	CU_JIT_INPUT_CUBIN = 0,
	CU_JIT_INPUT_PTX,
	CU_JIT_INPUT_FATBINARY,
	CU_JIT_INPUT_OBJECT,
	CU_JIT_INPUT_LIBRARY,
	CU_JIT_NUM_INPUT_TYPES
} CUjitInputType;
typedef struct CUlinkState_st* CUlinkState;
typedef enum CUgraphicsRegisterFlags_enum {
	CU_GRAPHICS_REGISTER_FLAGS_NONE = 0x00,
	CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 0x01,
	CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02,
	CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04,
	CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08
} CUgraphicsRegisterFlags;
typedef enum CUgraphicsMapResourceFlags_enum {
	CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00,
	CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01,
	CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
} CUgraphicsMapResourceFlags;
typedef enum CUarray_cubemap_face_enum {
	CU_CUBEMAP_FACE_POSITIVE_X = 0x00,
	CU_CUBEMAP_FACE_NEGATIVE_X = 0x01,
	CU_CUBEMAP_FACE_POSITIVE_Y = 0x02,
	CU_CUBEMAP_FACE_NEGATIVE_Y = 0x03,
	CU_CUBEMAP_FACE_POSITIVE_Z = 0x04,
	CU_CUBEMAP_FACE_NEGATIVE_Z = 0x05
} CUarray_cubemap_face;
typedef enum CUlimit_enum {
	CU_LIMIT_STACK_SIZE = 0x00,
	CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
	CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
	CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
	CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
	CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
	CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06,
	CU_LIMIT_MAX
} CUlimit;
typedef enum CUresourcetype_enum {
	CU_RESOURCE_TYPE_ARRAY = 0x00,
	CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
	CU_RESOURCE_TYPE_LINEAR = 0x02,
	CU_RESOURCE_TYPE_PITCH2D = 0x03
} CUresourcetype;
typedef void ( *CUhostFn)(void* userData);
typedef enum CUaccessProperty_enum {
	CU_ACCESS_PROPERTY_NORMAL = 0,
	CU_ACCESS_PROPERTY_STREAMING = 1,
	CU_ACCESS_PROPERTY_PERSISTING = 2
} CUaccessProperty;
typedef struct CUaccessPolicyWindow_st {
	void* base_ptr;
	size_t num_bytes;
	float hitRatio;
	CUaccessProperty hitProp;
	CUaccessProperty missProp;
} CUaccessPolicyWindow;
typedef struct CUDA_KERNEL_NODE_PARAMS_st {
	CUfunction func;
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;
	unsigned int sharedMemBytes;
	void** kernelParams;
	void** extra;
} CUDA_KERNEL_NODE_PARAMS;
typedef struct CUDA_MEMSET_NODE_PARAMS_st {
	CUdeviceptr dst;
	size_t pitch;
	unsigned int value;
	unsigned int elementSize;
	size_t width;
	size_t height;
} CUDA_MEMSET_NODE_PARAMS;
typedef struct CUDA_HOST_NODE_PARAMS_st {
	CUhostFn fn;
	void* userData;
} CUDA_HOST_NODE_PARAMS;
typedef enum CUgraphNodeType_enum {
	CU_GRAPH_NODE_TYPE_KERNEL = 0,
	CU_GRAPH_NODE_TYPE_MEMCPY = 1,
	CU_GRAPH_NODE_TYPE_MEMSET = 2,
	CU_GRAPH_NODE_TYPE_HOST = 3,
	CU_GRAPH_NODE_TYPE_GRAPH = 4,
	CU_GRAPH_NODE_TYPE_EMPTY = 5
} CUgraphNodeType;
typedef enum CUsynchronizationPolicy_enum {
	CU_SYNC_POLICY_AUTO = 1,
	CU_SYNC_POLICY_SPIN = 2,
	CU_SYNC_POLICY_YIELD = 3,
	CU_SYNC_POLICY_BLOCKING_SYNC = 4
} CUsynchronizationPolicy;
typedef enum CUkernelNodeAttrID_enum {
	CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
	CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2
} CUkernelNodeAttrID;
typedef union CUkernelNodeAttrValue_union {
	CUaccessPolicyWindow accessPolicyWindow;
	int cooperative;
} CUkernelNodeAttrValue;
typedef enum CUstreamCaptureStatus_enum {
	CU_STREAM_CAPTURE_STATUS_NONE = 0,
	CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
	CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
} CUstreamCaptureStatus;
typedef enum CUstreamCaptureMode_enum {
	CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
	CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
	CU_STREAM_CAPTURE_MODE_RELAXED = 2
} CUstreamCaptureMode;
typedef enum CUstreamAttrID_enum {
	CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
	CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
} CUstreamAttrID;
typedef union CUstreamAttrValue_union {
	CUaccessPolicyWindow accessPolicyWindow;
	CUsynchronizationPolicy syncPolicy;
} CUstreamAttrValue;
typedef enum cudaError_enum {
	CUDA_SUCCESS = 0,
	CUDA_ERROR_INVALID_VALUE = 1,
	CUDA_ERROR_OUT_OF_MEMORY = 2,
	CUDA_ERROR_NOT_INITIALIZED = 3,
	CUDA_ERROR_DEINITIALIZED = 4,
	CUDA_ERROR_PROFILER_DISABLED = 5,
	CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
	CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
	CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
	CUDA_ERROR_NO_DEVICE = 100,
	CUDA_ERROR_INVALID_DEVICE = 101,
	CUDA_ERROR_INVALID_IMAGE = 200,
	CUDA_ERROR_INVALID_CONTEXT = 201,
	CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
	CUDA_ERROR_MAP_FAILED = 205,
	CUDA_ERROR_UNMAP_FAILED = 206,
	CUDA_ERROR_ARRAY_IS_MAPPED = 207,
	CUDA_ERROR_ALREADY_MAPPED = 208,
	CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
	CUDA_ERROR_ALREADY_ACQUIRED = 210,
	CUDA_ERROR_NOT_MAPPED = 211,
	CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
	CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
	CUDA_ERROR_ECC_UNCORRECTABLE = 214,
	CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
	CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
	CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
	CUDA_ERROR_INVALID_PTX = 218,
	CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
	CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
	CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
	CUDA_ERROR_INVALID_SOURCE = 300,
	CUDA_ERROR_FILE_NOT_FOUND = 301,
	CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
	CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
	CUDA_ERROR_OPERATING_SYSTEM = 304,
	CUDA_ERROR_INVALID_HANDLE = 400,
	CUDA_ERROR_ILLEGAL_STATE = 401,
	CUDA_ERROR_NOT_FOUND = 500,
	CUDA_ERROR_NOT_READY = 600,
	CUDA_ERROR_ILLEGAL_ADDRESS = 700,
	CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
	CUDA_ERROR_LAUNCH_TIMEOUT = 702,
	CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
	CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
	CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
	CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
	CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
	CUDA_ERROR_ASSERT = 710,
	CUDA_ERROR_TOO_MANY_PEERS = 711,
	CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
	CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
	CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
	CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
	CUDA_ERROR_MISALIGNED_ADDRESS = 716,
	CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
	CUDA_ERROR_INVALID_PC = 718,
	CUDA_ERROR_LAUNCH_FAILED = 719,
	CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
	CUDA_ERROR_NOT_PERMITTED = 800,
	CUDA_ERROR_NOT_SUPPORTED = 801,
	CUDA_ERROR_SYSTEM_NOT_READY = 802,
	CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
	CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
	CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
	CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
	CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
	CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
	CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
	CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
	CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
	CUDA_ERROR_CAPTURED_EVENT = 907,
	CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
	CUDA_ERROR_TIMEOUT = 909,
	CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
	CUDA_ERROR_UNKNOWN = 999
} CUresult;
typedef enum CUdevice_P2PAttribute_enum {
	CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01,
	CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02,
	CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03,
	CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 0x04,
	CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 0x04
} CUdevice_P2PAttribute;
typedef void ( *CUstreamCallback)(CUstream hStream, CUresult status, void* userData);
typedef size_t ( *CUoccupancyB2DSize)(int blockSize);
typedef struct CUDA_MEMCPY2D_st {
	size_t srcXInBytes;
	size_t srcY;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	size_t srcPitch;
	size_t dstXInBytes;
	size_t dstY;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	size_t dstPitch;
	size_t WidthInBytes;
	size_t Height;
} CUDA_MEMCPY2D;
typedef struct CUDA_MEMCPY3D_st {
	size_t srcXInBytes;
	size_t srcY;
	size_t srcZ;
	size_t srcLOD;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	void* reserved0;
	size_t srcPitch;
	size_t srcHeight;
	size_t dstXInBytes;
	size_t dstY;
	size_t dstZ;
	size_t dstLOD;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	void* reserved1;
	size_t dstPitch;
	size_t dstHeight;
	size_t WidthInBytes;
	size_t Height;
	size_t Depth;
} CUDA_MEMCPY3D;
typedef struct CUDA_MEMCPY3D_PEER_st {
	size_t srcXInBytes;
	size_t srcY;
	size_t srcZ;
	size_t srcLOD;
	CUmemorytype srcMemoryType;
	const void* srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	CUcontext srcContext;
	size_t srcPitch;
	size_t srcHeight;
	size_t dstXInBytes;
	size_t dstY;
	size_t dstZ;
	size_t dstLOD;
	CUmemorytype dstMemoryType;
	void* dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	CUcontext dstContext;
	size_t dstPitch;
	size_t dstHeight;
	size_t WidthInBytes;
	size_t Height;
	size_t Depth;
} CUDA_MEMCPY3D_PEER;
typedef struct CUDA_ARRAY_DESCRIPTOR_st {
	size_t Width;
	size_t Height;
	CUarray_format Format;
	unsigned int NumChannels;
} CUDA_ARRAY_DESCRIPTOR;
typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
	size_t Width;
	size_t Height;
	size_t Depth;
	CUarray_format Format;
	unsigned int NumChannels;
	unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR;
typedef struct CUDA_RESOURCE_DESC_st {
	CUresourcetype resType;
	union {
		struct {
			CUarray hArray;
		} array;
		struct {
			CUmipmappedArray hMipmappedArray;
		} mipmap;
		struct {
			CUdeviceptr devPtr;
			CUarray_format format;
			unsigned int numChannels;
			size_t sizeInBytes;
		} linear;
		struct {
			CUdeviceptr devPtr;
			CUarray_format format;
			unsigned int numChannels;
			size_t width;
			size_t height;
			size_t pitchInBytes;
		} pitch2D;
		struct {
			int reserved[32];
		} reserved;
	} res;
	unsigned int flags;
} CUDA_RESOURCE_DESC;
typedef struct CUDA_TEXTURE_DESC_st {
	CUaddress_mode addressMode[3];
	CUfilter_mode filterMode;
	unsigned int flags;
	unsigned int maxAnisotropy;
	CUfilter_mode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
	float borderColor[4];
	int reserved[12];
} CUDA_TEXTURE_DESC;
typedef enum CUresourceViewFormat_enum {
	CU_RES_VIEW_FORMAT_NONE = 0x00,
	CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01,
	CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02,
	CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03,
	CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04,
	CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05,
	CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06,
	CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07,
	CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08,
	CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09,
	CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a,
	CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b,
	CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c,
	CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d,
	CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e,
	CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f,
	CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10,
	CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11,
	CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12,
	CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13,
	CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14,
	CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15,
	CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16,
	CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17,
	CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c,
	CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e,
	CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20,
	CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21,
	CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22
} CUresourceViewFormat;
typedef struct CUDA_RESOURCE_VIEW_DESC_st {
	CUresourceViewFormat format;
	size_t width;
	size_t height;
	size_t depth;
	unsigned int firstMipmapLevel;
	unsigned int lastMipmapLevel;
	unsigned int firstLayer;
	unsigned int lastLayer;
	unsigned int reserved[16];
} CUDA_RESOURCE_VIEW_DESC;
typedef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
	unsigned long long p2pToken;
	unsigned int vaSpaceToken;
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS;
typedef struct CUDA_LAUNCH_PARAMS_st {
	CUfunction function;
	unsigned int gridDimX;
	unsigned int gridDimY;
	unsigned int gridDimZ;
	unsigned int blockDimX;
	unsigned int blockDimY;
	unsigned int blockDimZ;
	unsigned int sharedMemBytes;
	CUstream hStream;
	void** kernelParams;
} CUDA_LAUNCH_PARAMS;
typedef enum CUexternalMemoryHandleType_enum {
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
	CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} CUexternalMemoryHandleType;
typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
	CUexternalMemoryHandleType type;
	union {
		int fd;
		struct {
			void* handle;
			const void* name;
		} win32;
		const void* nvSciBufObject;
	} handle;
	unsigned long long size;
	unsigned int flags;
	unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
	unsigned long long offset;
	unsigned long long size;
	unsigned int flags;
	unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC;
typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
	unsigned long long offset;
	CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
	unsigned int numLevels;
	unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;
typedef enum CUexternalSemaphoreHandleType_enum {
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
	CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
} CUexternalSemaphoreHandleType;
typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
	CUexternalSemaphoreHandleType type;
	union {
		int fd;
		struct {
			void* handle;
			const void* name;
		} win32;
		const void* nvSciSyncObj;
	} handle;
	unsigned int flags;
	unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;
typedef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
	struct {
		struct {
			unsigned long long value;
		} fence;
		union {
			void* fence;
			unsigned long long reserved;
		} nvSciSync;
		struct {
			unsigned long long key;
		} keyedMutex;
		unsigned int reserved[12];
	} params;
	unsigned int flags;
	unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
typedef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
	struct {
		struct {
			unsigned long long value;
		} fence;
		union {
			void* fence;
			unsigned long long reserved;
		} nvSciSync;
		struct {
			unsigned long long key;
			unsigned int timeoutMs;
		} keyedMutex;
		unsigned int reserved[10];
	} params;
	unsigned int flags;
	unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef enum CUmemAllocationHandleType_enum {
	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1,
	CU_MEM_HANDLE_TYPE_WIN32 = 0x2,
	CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4,
	CU_MEM_HANDLE_TYPE_MAX = 0xFFFFFFFF
} CUmemAllocationHandleType;
typedef enum CUmemAccess_flags_enum {
	CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
	CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
	CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3,
	CU_MEM_ACCESS_FLAGS_PROT_MAX = 0xFFFFFFFF
} CUmemAccess_flags;
typedef enum CUmemLocationType_enum {
	CU_MEM_LOCATION_TYPE_INVALID = 0x0,
	CU_MEM_LOCATION_TYPE_DEVICE = 0x1,
	CU_MEM_LOCATION_TYPE_MAX = 0xFFFFFFFF
} CUmemLocationType;
typedef enum CUmemAllocationType_enum {
	CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
	CU_MEM_ALLOCATION_TYPE_PINNED = 0x1,
	CU_MEM_ALLOCATION_TYPE_MAX = 0xFFFFFFFF
} CUmemAllocationType;
typedef enum CUmemAllocationGranularity_flags_enum {
	CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
	CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
} CUmemAllocationGranularity_flags;
typedef struct CUmemLocation_st {
	CUmemLocationType type;
	int id;
} CUmemLocation;
typedef enum CUmemAllocationCompType_enum {
	CU_MEM_ALLOCATION_COMP_NONE = 0x0,
	CU_MEM_ALLOCATION_COMP_GENERIC = 0x1
} CUmemAllocationCompType;
typedef struct CUmemAllocationProp_st {
	CUmemAllocationType type;
	CUmemAllocationHandleType requestedHandleTypes;
	CUmemLocation location;
	void* win32HandleMetaData;
	struct {
		unsigned char compressionType;
		unsigned char gpuDirectRDMACapable;
		unsigned char reserved[6];
	} allocFlags;
} CUmemAllocationProp;
typedef struct CUmemAccessDesc_st {
	CUmemLocation location;
	CUmemAccess_flags flags;
} CUmemAccessDesc;
typedef enum CUgraphExecUpdateResult_enum {
	CU_GRAPH_EXEC_UPDATE_SUCCESS = 0x0,
	CU_GRAPH_EXEC_UPDATE_ERROR = 0x1,
	CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 0x2,
	CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 0x3,
	CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 0x4,
	CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 0x5,
	CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 0x6
} CUgraphExecUpdateResult;
CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuInit(unsigned int Flags);
CUresult cuDriverGetVersion(int* driverVersion);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags);
__attribute__((deprecated)) CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev);
__attribute__((deprecated)) CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);
CUresult cuDevicePrimaryCtxReset(CUdevice dev);
CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxGetDevice(CUdevice* device);
CUresult cuCtxGetFlags(unsigned int* flags);
CUresult cuCtxSynchronize(void);
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit);
CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig);
CUresult cuCtxSetCacheConfig(CUfunc_cache config);
CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig);
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
CUresult cuCtxResetPersistingL2Cache(void);
__attribute__((deprecated)) CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags);
__attribute__((deprecated)) CUresult cuCtxDetach(CUcontext ctx);
CUresult cuModuleLoad(CUmodule* module, const char* fname);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
CUresult cuModuleUnload(CUmodule hmod);
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name);
CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name);
CUresult
cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
CUresult
cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name,
                 unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult
cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path,
                 unsigned int numOptions, CUjit_option* options, void** optionValues);
CUresult
cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut);
CUresult
cuLinkDestroy(CUlinkState state);
CUresult cuMemGetInfo(size_t* free, size_t* total);
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
CUresult cuMemAllocHost(void** pp, size_t bytesize);
CUresult cuMemFreeHost(void* p);
CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags);
CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p);
CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId);
CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev);
CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event);
CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle);
CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr);
CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags);
CUresult cuMemHostUnregister(void* p);
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy);
CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy);
CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy);
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream);
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream);
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
CUresult cuArrayCreate(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);
CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
CUresult cuArrayDestroy(CUarray hArray);
CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels);
CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);
CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr);
CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType);
CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr);
CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags);
CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
CUresult cuStreamGetPriority(CUstream hStream, int* priority);
CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags);
CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx);
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags);
CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode);
CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph);
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus);
CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus, cuuint64_t* id);
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
CUresult cuStreamQuery(CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuStreamDestroy(CUstream hStream);
CUresult cuStreamCopyAttributes(CUstream dst, CUstream src);
CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                              CUstreamAttrValue* value_out);
CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                              const CUstreamAttrValue* value);
CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuEventQuery(CUevent hEvent);
CUresult cuEventSynchronize(CUevent hEvent);
CUresult cuEventDestroy(CUevent hEvent);
CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc);
CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc);
CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc);
CUresult cuDestroyExternalMemory(CUexternalMemory extMem);
CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc);
CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream);
CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream);
CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags);
CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,
                        unsigned int gridDimY,
                        unsigned int gridDimZ,
                        unsigned int blockDimX,
                        unsigned int blockDimY,
                        unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra);
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned int gridDimX,
                                   unsigned int gridDimY,
                                   unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream,
                                   void** kernelParams);
CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags);
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData);
__attribute__((deprecated)) CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
__attribute__((deprecated)) CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
__attribute__((deprecated)) CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
__attribute__((deprecated)) CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
__attribute__((deprecated)) CUresult cuParamSetf(CUfunction hfunc, int offset, float value);
__attribute__((deprecated)) CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes);
__attribute__((deprecated)) CUresult cuLaunch(CUfunction f);
__attribute__((deprecated)) CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
__attribute__((deprecated)) CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
__attribute__((deprecated)) CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags);
CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams);
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams);
CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams);
CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams);
CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams);
CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams);
CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph);
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph);
CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies);
CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph);
CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type);
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes);
CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes);
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges);
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies);
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes);
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies);
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies);
CUresult cuGraphDestroyNode(CUgraphNode hNode);
CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize);
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
CUresult cuGraphDestroy(CUgraph hGraph);
CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out);
CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                       CUkernelNodeAttrValue* value_out);
CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                       const CUkernelNodeAttrValue* value);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
__attribute__((deprecated)) CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
__attribute__((deprecated)) CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
__attribute__((deprecated)) CUresult cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
__attribute__((deprecated)) CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch);
__attribute__((deprecated)) CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
__attribute__((deprecated)) CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
__attribute__((deprecated)) CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
__attribute__((deprecated)) CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
__attribute__((deprecated)) CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
__attribute__((deprecated)) CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
__attribute__((deprecated)) CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
__attribute__((deprecated)) CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor);
__attribute__((deprecated)) CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
__attribute__((deprecated)) CUresult cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim);
__attribute__((deprecated)) CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuTexRefCreate(CUtexref* pTexRef);
__attribute__((deprecated)) CUresult cuTexRefDestroy(CUtexref hTexRef);
__attribute__((deprecated)) CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
__attribute__((deprecated)) CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef);
CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
CUresult cuTexObjectDestroy(CUtexObject texObject);
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
CUresult cuSurfObjectDestroy(CUsurfObject surfObject);
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
CUresult cuCtxDisablePeerAccess(CUcontext peerContext);
CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);
CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags);
CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId);
CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc);
