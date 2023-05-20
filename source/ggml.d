module bert.ggml.ggml;

import core.stdc.config;

extern (C):
@nogc nothrow:

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_context * ctx = ggml_init(params);
//
//       struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//
//       ggml_set_param(ctx, x); // x is an input variable
//
//       struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
//       struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct ggml_cgraph gf = ggml_build_forward(f);
//
//       // set the input variable and parameter values
//       ggml_set_f32(x, 2.0f);
//       ggml_set_f32(a, 3.0f);
//       ggml_set_f32(b, 4.0f);
//
//       ggml_graph_compute(ctx0, &gf);
//
//       printf("f = %f\n", ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_graph_compute() function.
//
// The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - ggml_permute()
//   - ggml_conv_1d_1s()
//   - ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct ggml_tensor)
//
// The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_tensor * c = ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
//
//       // a[1, 2] = 1.0f;
//       *(float *) ((char *) a->data + 2*a->nb[1] + 1*a->nb[0]) = 1.0f;
//
//       // a[2, 0] = 2.0f;
//       *(float *) ((char *) a->data + 0*a->nb[1] + 2*a->nb[0]) = 2.0f;
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

enum GGML_FILE_MAGIC = 0x67676d6c; // "ggml"
enum GGML_FILE_VERSION = 1;

enum GGML_MAX_DIMS = 4;
enum GGML_MAX_NODES = 4096;
enum GGML_MAX_PARAMS = 16;
enum GGML_MAX_CONTEXTS = 64;
enum GGML_MAX_OPT = 4;
enum GGML_DEFAULT_N_THREADS = 4;

// we use the built-in 16-bit float type

alias ggml_fp16_t = ushort;

// convert FP16 <-> FP32
float ggml_fp16_to_fp32 (ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16 (float x);

void ggml_fp16_to_fp32_row (const(ggml_fp16_t)* x, float* y, size_t n);
void ggml_fp32_to_fp16_row (const(float)* x, ggml_fp16_t* y, size_t n);

struct ggml_context;

enum ggml_type
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4,
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_I8 = 10,
    GGML_TYPE_I16 = 11,
    GGML_TYPE_I32 = 12,
    GGML_TYPE_COUNT = 13
}

alias GGML_TYPE_F32 = ggml_type.GGML_TYPE_F32;
alias GGML_TYPE_F16 = ggml_type.GGML_TYPE_F16;
alias GGML_TYPE_Q4_0 = ggml_type.GGML_TYPE_Q4_0;
alias GGML_TYPE_Q4_1 = ggml_type.GGML_TYPE_Q4_1;
alias GGML_TYPE_Q4_2 = ggml_type.GGML_TYPE_Q4_2;
alias GGML_TYPE_Q5_0 = ggml_type.GGML_TYPE_Q5_0;
alias GGML_TYPE_Q5_1 = ggml_type.GGML_TYPE_Q5_1;
alias GGML_TYPE_Q8_0 = ggml_type.GGML_TYPE_Q8_0;
alias GGML_TYPE_Q8_1 = ggml_type.GGML_TYPE_Q8_1;
alias GGML_TYPE_I8 = ggml_type.GGML_TYPE_I8;
alias GGML_TYPE_I16 = ggml_type.GGML_TYPE_I16;
alias GGML_TYPE_I32 = ggml_type.GGML_TYPE_I32;
alias GGML_TYPE_COUNT = ggml_type.GGML_TYPE_COUNT;

// model file types
enum ggml_ftype
{
    GGML_FTYPE_UNKNOWN = -1,
    GGML_FTYPE_ALL_F32 = 0,
    GGML_FTYPE_MOSTLY_F16 = 1, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_0 = 2, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1 = 3, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    GGML_FTYPE_MOSTLY_Q4_2 = 5, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q8_0 = 7, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_0 = 8, // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_1 = 9 // except 1d tensors
}

alias GGML_FTYPE_UNKNOWN = ggml_ftype.GGML_FTYPE_UNKNOWN;
alias GGML_FTYPE_ALL_F32 = ggml_ftype.GGML_FTYPE_ALL_F32;
alias GGML_FTYPE_MOSTLY_F16 = ggml_ftype.GGML_FTYPE_MOSTLY_F16;
alias GGML_FTYPE_MOSTLY_Q4_0 = ggml_ftype.GGML_FTYPE_MOSTLY_Q4_0;
alias GGML_FTYPE_MOSTLY_Q4_1 = ggml_ftype.GGML_FTYPE_MOSTLY_Q4_1;
alias GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = ggml_ftype.GGML_FTYPE_MOSTLY_Q4_1_SOME_F16;
alias GGML_FTYPE_MOSTLY_Q4_2 = ggml_ftype.GGML_FTYPE_MOSTLY_Q4_2;
alias GGML_FTYPE_MOSTLY_Q8_0 = ggml_ftype.GGML_FTYPE_MOSTLY_Q8_0;
alias GGML_FTYPE_MOSTLY_Q5_0 = ggml_ftype.GGML_FTYPE_MOSTLY_Q5_0;
alias GGML_FTYPE_MOSTLY_Q5_1 = ggml_ftype.GGML_FTYPE_MOSTLY_Q5_1;

// available tensor operations:
enum ggml_op
{
    GGML_OP_NONE = 0,

    GGML_OP_DUP = 1,
    GGML_OP_ADD = 2,
    GGML_OP_SUB = 3,
    GGML_OP_MUL = 4,
    GGML_OP_DIV = 5,
    GGML_OP_SQR = 6,
    GGML_OP_SQRT = 7,
    GGML_OP_SUM = 8,
    GGML_OP_MEAN = 9,
    GGML_OP_REPEAT = 10,
    GGML_OP_ABS = 11,
    GGML_OP_SGN = 12,
    GGML_OP_NEG = 13,
    GGML_OP_STEP = 14,
    GGML_OP_RELU = 15,
    GGML_OP_GELU = 16,
    GGML_OP_SILU = 17,
    GGML_OP_NORM = 18, // normalize
    GGML_OP_RMS_NORM = 19,

    GGML_OP_MUL_MAT = 20,

    GGML_OP_SCALE = 21,
    GGML_OP_CPY = 22,
    GGML_OP_CONT = 23,
    GGML_OP_RESHAPE = 24,
    GGML_OP_VIEW = 25,
    GGML_OP_PERMUTE = 26,
    GGML_OP_TRANSPOSE = 27,
    GGML_OP_GET_ROWS = 28,
    GGML_OP_DIAG_MASK_INF = 29,
    GGML_OP_SOFT_MAX = 30,
    GGML_OP_ROPE = 31,
    GGML_OP_ALIBI = 32,
    GGML_OP_CONV_1D_1S = 33,
    GGML_OP_CONV_1D_2S = 34,

    GGML_OP_FLASH_ATTN = 35,
    GGML_OP_FLASH_FF = 36,

    GGML_OP_MAP_UNARY = 37,
    GGML_OP_MAP_BINARY = 38,

    GGML_OP_COUNT = 39
}

alias GGML_OP_NONE = ggml_op.GGML_OP_NONE;
alias GGML_OP_DUP = ggml_op.GGML_OP_DUP;
alias GGML_OP_ADD = ggml_op.GGML_OP_ADD;
alias GGML_OP_SUB = ggml_op.GGML_OP_SUB;
alias GGML_OP_MUL = ggml_op.GGML_OP_MUL;
alias GGML_OP_DIV = ggml_op.GGML_OP_DIV;
alias GGML_OP_SQR = ggml_op.GGML_OP_SQR;
alias GGML_OP_SQRT = ggml_op.GGML_OP_SQRT;
alias GGML_OP_SUM = ggml_op.GGML_OP_SUM;
alias GGML_OP_MEAN = ggml_op.GGML_OP_MEAN;
alias GGML_OP_REPEAT = ggml_op.GGML_OP_REPEAT;
alias GGML_OP_ABS = ggml_op.GGML_OP_ABS;
alias GGML_OP_SGN = ggml_op.GGML_OP_SGN;
alias GGML_OP_NEG = ggml_op.GGML_OP_NEG;
alias GGML_OP_STEP = ggml_op.GGML_OP_STEP;
alias GGML_OP_RELU = ggml_op.GGML_OP_RELU;
alias GGML_OP_GELU = ggml_op.GGML_OP_GELU;
alias GGML_OP_SILU = ggml_op.GGML_OP_SILU;
alias GGML_OP_NORM = ggml_op.GGML_OP_NORM;
alias GGML_OP_RMS_NORM = ggml_op.GGML_OP_RMS_NORM;
alias GGML_OP_MUL_MAT = ggml_op.GGML_OP_MUL_MAT;
alias GGML_OP_SCALE = ggml_op.GGML_OP_SCALE;
alias GGML_OP_CPY = ggml_op.GGML_OP_CPY;
alias GGML_OP_CONT = ggml_op.GGML_OP_CONT;
alias GGML_OP_RESHAPE = ggml_op.GGML_OP_RESHAPE;
alias GGML_OP_VIEW = ggml_op.GGML_OP_VIEW;
alias GGML_OP_PERMUTE = ggml_op.GGML_OP_PERMUTE;
alias GGML_OP_TRANSPOSE = ggml_op.GGML_OP_TRANSPOSE;
alias GGML_OP_GET_ROWS = ggml_op.GGML_OP_GET_ROWS;
alias GGML_OP_DIAG_MASK_INF = ggml_op.GGML_OP_DIAG_MASK_INF;
alias GGML_OP_SOFT_MAX = ggml_op.GGML_OP_SOFT_MAX;
alias GGML_OP_ROPE = ggml_op.GGML_OP_ROPE;
alias GGML_OP_ALIBI = ggml_op.GGML_OP_ALIBI;
alias GGML_OP_CONV_1D_1S = ggml_op.GGML_OP_CONV_1D_1S;
alias GGML_OP_CONV_1D_2S = ggml_op.GGML_OP_CONV_1D_2S;
alias GGML_OP_FLASH_ATTN = ggml_op.GGML_OP_FLASH_ATTN;
alias GGML_OP_FLASH_FF = ggml_op.GGML_OP_FLASH_FF;
alias GGML_OP_MAP_UNARY = ggml_op.GGML_OP_MAP_UNARY;
alias GGML_OP_MAP_BINARY = ggml_op.GGML_OP_MAP_BINARY;
alias GGML_OP_COUNT = ggml_op.GGML_OP_COUNT;

// ggml object
struct ggml_object
{
    alias size_t = c_ulong;
    size_t offs;
    size_t size;

    ggml_object* next;

    char[8] padding;
}

extern __gshared const size_t GGML_OBJECT_SIZE;

// n-dimensional tensor
struct ggml_tensor
{
    ggml_type type;

    int n_dims;
    long[GGML_MAX_DIMS] ne; // number of elements
    size_t[GGML_MAX_DIMS] nb; // stride in bytes:
    // nb[0] = sizeof(type)
    // nb[1] = nb[0]   * ne[0] + padding
    // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    ggml_op op;

    bool is_param;

    ggml_tensor* grad;
    ggml_tensor* src0;
    ggml_tensor* src1;
    ggml_tensor*[GGML_MAX_OPT] opt;

    // thread scheduling
    int n_tasks;

    // performance
    int perf_runs;
    long perf_cycles;
    long perf_time_us;

    void* data;

    char[32] name;

    char[8] padding; // TODO: remove and add padding to name?
}

// computation graph
struct ggml_cgraph
{
    int n_nodes;
    int n_leafs;
    int n_threads;

    size_t work_size;
    ggml_tensor* work;

    ggml_tensor*[GGML_MAX_NODES] nodes;
    ggml_tensor*[GGML_MAX_NODES] grads;
    ggml_tensor*[GGML_MAX_NODES] leafs;

    // performance
    int perf_runs;
    long perf_cycles;
    long perf_time_us;
}

// scratch buffer
struct ggml_scratch
{
    size_t offs;
    size_t size;
    void* data;
}

struct ggml_init_params
{
    // memory pool
    size_t mem_size; // bytes
    void* mem_buffer; // if NULL, memory will be allocated internally
    bool no_alloc; // don't allocate memory for the tensor data
}

// misc

void ggml_time_init (); // call this once at the beginning of the program
long ggml_time_ms ();
long ggml_time_us ();
long ggml_cycles ();
long ggml_cycles_per_ms ();

void ggml_print_object (const(ggml_object)* obj);
void ggml_print_objects (const(ggml_context)* ctx);

long ggml_nelements (const(ggml_tensor)* tensor);
size_t ggml_nbytes (const(ggml_tensor)* tensor);

int ggml_blck_size (ggml_type type);
size_t ggml_type_size (ggml_type type); // size in bytes for all elements in a block
float ggml_type_sizef (ggml_type type); // ggml_type_size()/ggml_blck_size() as float

const(char)* ggml_type_name (ggml_type type);

size_t ggml_element_size (const(ggml_tensor)* tensor);

bool ggml_is_quantized (ggml_type type);

// TODO: temporary until model loading of ggml examples is refactored
ggml_type ggml_ftype_to_ggml_type (ggml_ftype ftype);

// main

ggml_context* ggml_init (ggml_init_params params);
void ggml_free (ggml_context* ctx);

size_t ggml_used_mem (const(ggml_context)* ctx);

size_t ggml_set_scratch (ggml_context* ctx, ggml_scratch scratch);

ggml_tensor* ggml_new_tensor (
    ggml_context* ctx,
    ggml_type type,
    int n_dims,
    const(long)* ne);

ggml_tensor* ggml_new_tensor_1d (ggml_context* ctx, ggml_type type, long ne0);

ggml_tensor* ggml_new_tensor_2d (
    ggml_context* ctx,
    ggml_type type,
    long ne0,
    long ne1);

ggml_tensor* ggml_new_tensor_3d (
    ggml_context* ctx,
    ggml_type type,
    long ne0,
    long ne1,
    long ne2);

ggml_tensor* ggml_new_tensor_4d (
    ggml_context* ctx,
    ggml_type type,
    long ne0,
    long ne1,
    long ne2,
    long ne3);

ggml_tensor* ggml_new_i32 (ggml_context* ctx, int value);
ggml_tensor* ggml_new_f32 (ggml_context* ctx, float value);

ggml_tensor* ggml_dup_tensor (ggml_context* ctx, const(ggml_tensor)* src);
ggml_tensor* ggml_view_tensor (ggml_context* ctx, const(ggml_tensor)* src);

ggml_tensor* ggml_set_zero (ggml_tensor* tensor);
ggml_tensor* ggml_set_i32 (ggml_tensor* tensor, int value);
ggml_tensor* ggml_set_f32 (ggml_tensor* tensor, float value);

int ggml_get_i32_1d (const(ggml_tensor)* tensor, int i);
void ggml_set_i32_1d (const(ggml_tensor)* tensor, int i, int value);

float ggml_get_f32_1d (const(ggml_tensor)* tensor, int i);
void ggml_set_f32_1d (const(ggml_tensor)* tensor, int i, float value);

void* ggml_get_data (const(ggml_tensor)* tensor);
float* ggml_get_data_f32 (const(ggml_tensor)* tensor);

const(char)* ggml_get_name (const(ggml_tensor)* tensor);
void ggml_set_name (ggml_tensor* tensor, const(char)* name);

//
// operations on tensors with backpropagation
//

ggml_tensor* ggml_dup (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_add (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_add_inplace (
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b);

ggml_tensor* ggml_sub (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_mul (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_div (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_sqr (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_sqrt (ggml_context* ctx, ggml_tensor* a);

// return scalar
// TODO: compute sum along rows
ggml_tensor* ggml_sum (ggml_context* ctx, ggml_tensor* a);

// mean along rows
ggml_tensor* ggml_mean (ggml_context* ctx, ggml_tensor* a);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
ggml_tensor* ggml_repeat (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_abs (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_sgn (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_neg (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_step (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_relu (ggml_context* ctx, ggml_tensor* a);

// TODO: double-check this computation is correct
ggml_tensor* ggml_gelu (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_silu (ggml_context* ctx, ggml_tensor* a);

// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
ggml_tensor* ggml_norm (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_rms_norm (ggml_context* ctx, ggml_tensor* a);

// A: m rows, n columns
// B: p rows, n columns (i.e. we transpose it internally)
// result is m columns, p rows
ggml_tensor* ggml_mul_mat (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

//
// operations on tensors without backpropagation
//

// in-place, returns view(a)
ggml_tensor* ggml_scale (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

// a -> b, return view(b)
ggml_tensor* ggml_cpy (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

// make contiguous
ggml_tensor* ggml_cont (ggml_context* ctx, ggml_tensor* a);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
ggml_tensor* ggml_reshape (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
ggml_tensor* ggml_reshape_2d (
    ggml_context* ctx,
    ggml_tensor* a,
    long ne0,
    long ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
ggml_tensor* ggml_reshape_3d (
    ggml_context* ctx,
    ggml_tensor* a,
    long ne0,
    long ne1,
    long ne2);

// offset in bytes
ggml_tensor* ggml_view_1d (
    ggml_context* ctx,
    ggml_tensor* a,
    long ne0,
    size_t offset);

// row stride in bytes
ggml_tensor* ggml_view_2d (
    ggml_context* ctx,
    ggml_tensor* a,
    long ne0,
    long ne1,
    size_t nb1,
    size_t offset);

// row   stride in bytes
// slice stride in bytes
ggml_tensor* ggml_view_3d (
    ggml_context* ctx,
    ggml_tensor* a,
    long ne0,
    long ne1,
    long ne2,
    size_t nb1,
    size_t nb2,
    size_t offset);

ggml_tensor* ggml_permute (
    ggml_context* ctx,
    ggml_tensor* a,
    int axis0,
    int axis1,
    int axis2,
    int axis3);

// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
ggml_tensor* ggml_transpose (ggml_context* ctx, ggml_tensor* a);

ggml_tensor* ggml_get_rows (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

// set elements above the diagonal to -INF
// in-place, returns view(a)
ggml_tensor* ggml_diag_mask_inf (ggml_context* ctx, ggml_tensor* a, int n_past);

// in-place, returns view(a)
ggml_tensor* ggml_soft_max (ggml_context* ctx, ggml_tensor* a);

// rotary position embedding
// in-place, returns view(a)
// if mode & 1 == 1, skip n_past elements
// if mode & 2 == 1, GPT-NeoX style
// TODO: avoid creating a new tensor every time
ggml_tensor* ggml_rope (
    ggml_context* ctx,
    ggml_tensor* a,
    int n_past,
    int n_dims,
    int mode);

// alibi position embedding
// in-place, returns view(a)
ggml_tensor* ggml_alibi (
    ggml_context* ctx,
    ggml_tensor* a,
    int n_past,
    int n_head);

// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
ggml_tensor* ggml_conv_1d_1s (
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b);

ggml_tensor* ggml_conv_1d_2s (
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b);

ggml_tensor* ggml_flash_attn (
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* k,
    ggml_tensor* v,
    bool masked);

ggml_tensor* ggml_flash_ff (
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b0,
    ggml_tensor* b1,
    ggml_tensor* c0,
    ggml_tensor* c1);

// Mapping operations
alias ggml_unary_op_f32_t = void function (const int, float*, const(float)*);
alias ggml_binary_op_f32_t = void function (const int, float*, const(float)*, const(float)*);

ggml_tensor* ggml_map_unary_f32 (
    ggml_context* ctx,
    ggml_tensor* a,
    const ggml_unary_op_f32_t fun);

ggml_tensor* ggml_map_binary_f32 (
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    const ggml_binary_op_f32_t fun);

//
// automatic differentiation
//

void ggml_set_param (ggml_context* ctx, ggml_tensor* tensor);

void ggml_build_forward_expand (ggml_cgraph* cgraph, ggml_tensor* tensor);

ggml_cgraph ggml_build_forward (ggml_tensor* tensor);
ggml_cgraph ggml_build_backward (ggml_context* ctx, ggml_cgraph* gf, bool keep);

void ggml_graph_compute (ggml_context* ctx, ggml_cgraph* cgraph);
void ggml_graph_reset (ggml_cgraph* cgraph);

// print info and performance information for the graph
void ggml_graph_print (const(ggml_cgraph)* cgraph);

// dump the graph into a file using the dot format
void ggml_graph_dump_dot (const(ggml_cgraph)* gb, const(ggml_cgraph)* gf, const(char)* filename);

//
// optimization
//

// optimization methods
enum ggml_opt_type
{
    GGML_OPT_ADAM = 0,
    GGML_OPT_LBFGS = 1
}

alias GGML_OPT_ADAM = ggml_opt_type.GGML_OPT_ADAM;
alias GGML_OPT_LBFGS = ggml_opt_type.GGML_OPT_LBFGS;

// linesearch methods
enum ggml_linesearch
{
    GGML_LINESEARCH_DEFAULT = 1,

    GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0,
    GGML_LINESEARCH_BACKTRACKING_WOLFE = 1,
    GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2
}

alias GGML_LINESEARCH_DEFAULT = ggml_linesearch.GGML_LINESEARCH_DEFAULT;
alias GGML_LINESEARCH_BACKTRACKING_ARMIJO = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_ARMIJO;
alias GGML_LINESEARCH_BACKTRACKING_WOLFE = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_WOLFE;
alias GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = ggml_linesearch.GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

// optimization return values
enum ggml_opt_result
{
    GGML_OPT_OK = 0,
    GGML_OPT_DID_NOT_CONVERGE = 1,
    GGML_OPT_NO_CONTEXT = 2,
    GGML_OPT_INVALID_WOLFE = 3,
    GGML_OPT_FAIL = 4,

    GGML_LINESEARCH_FAIL = -128,
    GGML_LINESEARCH_MINIMUM_STEP = -127,
    GGML_LINESEARCH_MAXIMUM_STEP = -126,
    GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125,
    GGML_LINESEARCH_INVALID_PARAMETERS = -124
}

alias GGML_OPT_OK = ggml_opt_result.GGML_OPT_OK;
alias GGML_OPT_DID_NOT_CONVERGE = ggml_opt_result.GGML_OPT_DID_NOT_CONVERGE;
alias GGML_OPT_NO_CONTEXT = ggml_opt_result.GGML_OPT_NO_CONTEXT;
alias GGML_OPT_INVALID_WOLFE = ggml_opt_result.GGML_OPT_INVALID_WOLFE;
alias GGML_OPT_FAIL = ggml_opt_result.GGML_OPT_FAIL;
alias GGML_LINESEARCH_FAIL = ggml_opt_result.GGML_LINESEARCH_FAIL;
alias GGML_LINESEARCH_MINIMUM_STEP = ggml_opt_result.GGML_LINESEARCH_MINIMUM_STEP;
alias GGML_LINESEARCH_MAXIMUM_STEP = ggml_opt_result.GGML_LINESEARCH_MAXIMUM_STEP;
alias GGML_LINESEARCH_MAXIMUM_ITERATIONS = ggml_opt_result.GGML_LINESEARCH_MAXIMUM_ITERATIONS;
alias GGML_LINESEARCH_INVALID_PARAMETERS = ggml_opt_result.GGML_LINESEARCH_INVALID_PARAMETERS;

// optimization parameters
//
//   see ggml.c (ggml_opt_default_params) for default values
//
struct ggml_opt_params
{
    ggml_opt_type type;

    int n_threads;

    // delta-based convergence test
    //
    //   if past == 0 - disabled
    //   if past > 0:
    //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
    //
    int past;
    float delta;

    // maximum number of iterations without improvement
    //
    //   if 0 - disabled
    //   if > 0:
    //     assume convergence if no cost improvement in this number of iterations
    //
    int max_no_improvement;

    bool print_forward_graph;
    bool print_backward_graph;

    // ADAM parameters

    // learning rate

    // epsilon for numerical stability
    // epsilon for convergence test
    // epsilon for convergence test
    struct _Anonymous_0
    {
        int n_iter;
        float alpha;
        float beta1;
        float beta2;
        float eps;
        float eps_f;
        float eps_g;
    }

    _Anonymous_0 adam;

    // LBFGS parameters

    // number of corrections to approximate the inv. Hessian

    // convergence tolerance
    // line search tolerance
    struct _Anonymous_1
    {
        int m;
        int n_iter;
        int max_linesearch;
        float eps;
        float ftol;
        float wolfe;
        float min_step;
        float max_step;
        ggml_linesearch linesearch;
    }

    _Anonymous_1 lbfgs;
}

ggml_opt_params ggml_opt_default_params (ggml_opt_type type);

// optimize the function defined by the tensor f
ggml_opt_result ggml_opt (
    ggml_context* ctx,
    ggml_opt_params params,
    ggml_tensor* f);

//
// quantization
//

size_t ggml_quantize_q4_0 (const(float)* src, void* dst, int n, int k, long* hist);
size_t ggml_quantize_q4_1 (const(float)* src, void* dst, int n, int k, long* hist);
size_t ggml_quantize_q4_2 (const(float)* src, void* dst, int n, int k, long* hist);
size_t ggml_quantize_q5_0 (const(float)* src, void* dst, int n, int k, long* hist);
size_t ggml_quantize_q5_1 (const(float)* src, void* dst, int n, int k, long* hist);
size_t ggml_quantize_q8_0 (const(float)* src, void* dst, int n, int k, long* hist);

size_t ggml_quantize_chunk (ggml_type type, const(float)* src, void* dst, int start, int n, long* hist);

//
// system info
//

int ggml_cpu_has_avx ();
int ggml_cpu_has_avx2 ();
int ggml_cpu_has_avx512 ();
int ggml_cpu_has_avx512_vbmi ();
int ggml_cpu_has_avx512_vnni ();
int ggml_cpu_has_fma ();
int ggml_cpu_has_neon ();
int ggml_cpu_has_arm_fma ();
int ggml_cpu_has_f16c ();
int ggml_cpu_has_fp16_va ();
int ggml_cpu_has_wasm_simd ();
int ggml_cpu_has_blas ();
int ggml_cpu_has_cublas ();
int ggml_cpu_has_clblast ();
int ggml_cpu_has_gpublas ();
int ggml_cpu_has_sse3 ();
int ggml_cpu_has_vsx ();

//
// Internal types and functions exposed for tests and benchmarks
//

// restrict not standard in C++

alias dequantize_row_q_t = void function (const(void)* x, float* y, int k);
alias quantize_row_q_t = void function (const(float)* x, void* y, int k);
alias vec_dot_q_t = void function (const int n, float* s, const(void)* x, const(void)* y);

struct quantize_fns_t
{
    dequantize_row_q_t dequantize_row_q;
    quantize_row_q_t quantize_row_q;
    quantize_row_q_t quantize_row_q_reference;
    quantize_row_q_t quantize_row_q_dot;
    vec_dot_q_t vec_dot_q;
    ggml_type vec_dot_type;
}

quantize_fns_t ggml_internal_get_quantize_fn (size_t i);

