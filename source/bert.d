module bert.bert.bert;

extern (C) @nogc nothrow:

// struct bert_params
// {
//     int32_t n_threads = 6;
//     int32_t port = 8080; // server mode port to bind

//     const char* model = "models/all-MiniLM-L6-v2/ggml-model-q4_0.bin"; // model path
//     const char* prompt = "test prompt";
// };

// bool bert_params_parse(int argc, char **argv, bert_params &params);

struct bert_ctx;

alias bert_vocab_id = int;

bert_ctx* bert_load_from_file (const(char)* fname);
void bert_free (bert_ctx* ctx);

// Main api, does both tokenizing and evaluation

void bert_encode (
    bert_ctx* ctx,
    int n_threads,
    const(char)* texts,
    float* embeddings);

// n_batch_size - how many to process at a time
// n_inputs     - total size of texts and embeddings arrays
void bert_encode_batch (
    bert_ctx* ctx,
    int n_threads,
    int n_batch_size,
    int n_inputs,
    const(char*)* texts,
    float** embeddings);

// Api for separate tokenization & eval

void bert_tokenize (
    bert_ctx* ctx,
    const(char)* text,
    bert_vocab_id* tokens,
    int* n_tokens,
    int n_max_tokens);

void bert_eval (
    bert_ctx* ctx,
    int n_threads,
    bert_vocab_id* tokens,
    int n_tokens,
    float* embeddings);

// NOTE: for batch processing the longest input must be first
void bert_eval_batch (
    bert_ctx* ctx,
    int n_threads,
    int n_batch_size,
    bert_vocab_id** batch_tokens,
    int* n_tokens,
    float** batch_embeddings);

int bert_n_embd (bert_ctx* ctx);
int bert_n_max_tokens (bert_ctx* ctx);

const(char)* bert_vocab_id_to_token (bert_ctx* ctx, bert_vocab_id id);

// BERT_H
