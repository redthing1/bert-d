import std.stdio;
import std.getopt;
import std.string;
import std.conv;

import bert.bert;
import bert.ggml;
import std.container.array;

struct Options {
	string model;
	string prompt;
	int n_threads = 0;
}

int main(string[] args) {
	enum __func__ = "main";
	const long t_main_start_us = ggml_time_us();

	Options params;
	params.model = "../../models/all-MiniLM-L6-v2/ggml-model-f32.bin";
	params.prompt = "test prompt";

	auto help_info = getopt(args, "m", &params.model, "p", &params.prompt, "n", &params.n_threads);
	if (help_info.helpWanted) {
		defaultGetoptPrinter("bert-d example", help_info.options);
		return 0;
	}

	long t_load_us = 0;

	bert_ctx* bctx;

	// load the model
	{
		const long t_start_us = ggml_time_us();

		if ((bctx = bert_load_from_file(params.model.toStringz)) == null) {
			// fwritef(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
			stderr.writef("%s: failed to load model from '%s'\n", __func__, params.model);
			return 1;
		}

		t_load_us = ggml_time_us() - t_start_us;
	}

	long t_eval_us = 0;
	long t_start_us = ggml_time_us();
	int N = bert_n_max_tokens(bctx);
	// tokenize the prompt
	// Array!bert_vocab_id tokens(N);
	Array!bert_vocab_id tokens;
	tokens.reserve(N);
	int n_tokens;
	bert_tokenize(bctx, params.prompt.toStringz, cast(int*) tokens.data, &n_tokens, N);
	// tokens.resize(n_tokens);
	tokens.length = n_tokens;

	writef("%s: number of tokens in prompt = %d\n", __func__, tokens.length);
	writef("\n");

	writef("[");
	foreach (tok; tokens) {
		writef("%d, ", tok);
	}
	writef("]\n");

	foreach (tok; tokens) {
		writef("%d -> %s\n", tok, bert_vocab_id_to_token(bctx, tok).to!string);
	}
	// Array!float embeddings(bert_n_embd(bctx));
	Array!float embeddings;
	embeddings.reserve(bert_n_embd(bctx));

	bert_eval(bctx, params.n_threads, cast(int*) tokens.data, n_tokens, cast(float*) embeddings.data);
	t_eval_us += ggml_time_us() - t_start_us;

	writef("[");
	foreach (e; embeddings) {
		writef("%1.4f, ", e);
	}
	writef("]\n");

	// report timing
	{
		const long t_main_end_us = ggml_time_us();

		writef("\n\n");
		//writef("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
		writef("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
		writef("%s:  eval time = %8.2f ms / %.2f ms per token\n", __func__,
			t_eval_us / 1000.0f, t_eval_us / 1000.0f / tokens.length);
		writef("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
	}

	return 0;
}
