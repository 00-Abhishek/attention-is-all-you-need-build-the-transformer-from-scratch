"""
Attention Is All You Need: Build the Transformer From Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of a from-scratch Transformer: build vocab, forward pass,
a couple of training steps, and greedy/argmax next-token selection."""

import numpy as np
import torch

from solution import (
    build_token_to_id_vocab, build_id_to_token_vocab,
    encode_sentence_to_ids, decode_ids_to_tokens,
    pad_id_sequence, stack_padded_sequences_to_batch,
    scale_embeddings_by_sqrt_d_model,
    compute_positional_div_term, build_position_index_column,
    fill_even_indices_with_sin, fill_odd_indices_with_cos,
    build_sinusoidal_positional_encoding, add_positional_encoding_to_embeddings,
    build_padding_mask, build_causal_mask, combine_padding_and_causal_masks,
    compute_raw_attention_scores, scale_attention_scores,
    mask_attention_scores_with_neg_inf, softmax_attention_weights,
    apply_attention_weights_to_values, scaled_dot_product_attention,
    split_last_dim_into_heads, transpose_heads_before_sequence,
    merge_heads_back_to_model_dim, apply_linear_projection,
    project_to_query_key_value, split_qkv_into_heads,
    multi_head_scaled_dot_product_attention, merge_heads_and_project_output,
    assemble_multi_head_attention_forward,
    apply_ffn_first_linear_and_relu, apply_ffn_second_linear,
    position_wise_feed_forward_network,
    compute_layer_norm_mean_and_variance, normalize_and_scale_with_gamma_beta,
    apply_residual_add_and_norm, apply_dropout_with_keep_mask,
    encoder_layer_self_attention_sublayer, encoder_layer_feed_forward_sublayer,
    assemble_encoder_layer, stack_encoder_layers,
    decoder_layer_masked_self_attention_sublayer,
    decoder_layer_cross_attention_sublayer,
    decoder_layer_feed_forward_sublayer,
    assemble_decoder_layer, stack_decoder_layers,
    apply_final_output_projection, tie_output_projection_to_token_embeddings,
    apply_log_softmax_over_vocab, run_transformer_forward,
    init_encoder_layer_parameters, init_decoder_layer_parameters,
    init_embedding_and_projection_parameters, collect_model_parameters_into_list,
    shift_targets_right_with_start_token, compute_noam_learning_rate,
    build_uniform_smoothing_distribution, set_confidence_on_gold_tokens,
    zero_pad_column_and_pad_token_rows, compute_label_smoothed_kl_loss,
    average_loss_over_non_pad_tokens, compute_token_accuracy_ignoring_pad,
    compute_batch_training_loss, run_training_step_with_backprop,
    run_training_loop_for_steps,
    initialize_adam_optimizer_state, update_adam_first_moment,
    update_adam_second_moment, apply_adam_bias_correction,
    compute_adam_parameter_update, apply_adam_step_to_all_parameters,
    zero_all_parameter_gradients,
    pick_next_token_by_argmax, compute_length_penalty,
    compute_candidate_scores, select_top_k_candidates,
    append_tokens_to_beam_sequences, mark_finished_beams,
    select_best_finished_beam,
)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- 1. Build a tiny toy parallel corpus and vocab ----
    src_sentences = ["hello world", "good morning"]
    tgt_sentences = ["bonjour monde", "bon matin"]
    all_sents = src_sentences + tgt_sentences

    tok2id = build_token_to_id_vocab(all_sents)
    id2tok = build_id_to_token_vocab(tok2id)
    pad_id = tok2id["<pad>"]
    bos_id = tok2id["<bos>"]
    eos_id = tok2id["<eos>"]
    vocab_size = len(tok2id)
    print(f"Vocab size: {vocab_size}; pad={pad_id}, bos={bos_id}, eos={eos_id}")

    # ---- 2. Encode + pad + batch ----
    max_len = 4
    src_batch = stack_padded_sequences_to_batch(
        [pad_id_sequence(encode_sentence_to_ids(s, tok2id) + [eos_id], max_len, pad_id)
         for s in src_sentences])
    tgt_batch = stack_padded_sequences_to_batch(
        [pad_id_sequence([bos_id] + encode_sentence_to_ids(s, tok2id) + [eos_id], max_len, pad_id)
         for s in tgt_sentences])
    print("src_batch shape:", tuple(src_batch.shape), "tgt_batch shape:", tuple(tgt_batch.shape))

    # ---- 3. Initialize tiny model parameters ----
    d_model, num_heads, d_ff, num_layers = 16, 2, 32, 2
    enc_layers = [init_encoder_layer_parameters(d_model, num_heads, d_ff) for _ in range(num_layers)]
    dec_layers = [init_decoder_layer_parameters(d_model, num_heads, d_ff) for _ in range(num_layers)]
    emb_params = init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True)
    model_params = {
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "embeddings": emb_params,
        "token_embedding": emb_params["src_embedding"],
        "output_projection": emb_params["output_projection"],
        "d_model": d_model,
    }
    parameter_list = collect_model_parameters_into_list(enc_layers, dec_layers, emb_params)
    print(f"Total parameter tensors: {len(parameter_list)}")

    # ---- 4. One forward pass for shape sanity ----
    log_probs = run_transformer_forward(src_batch, tgt_batch, model_params, num_heads, pad_id)
    print("log_probs shape:", tuple(log_probs.shape))

    # ---- 5. A few training steps on the toy batch and watch loss decrease ----
    config = {
        "num_heads": num_heads, "pad_id": pad_id, "bos_id": bos_id,
        "start_id": bos_id,
        "d_model": d_model, "warmup_steps": 50,
        "label_smoothing": 0.1, "smoothing": 0.1,
        "vocab_size": vocab_size,
    }
    optim_state = initialize_adam_optimizer_state(parameter_list)
    batches = [(src_batch, tgt_batch)] * 6
    losses = run_training_loop_for_steps(batches, parameter_list, model_params,
                                         optim_state, num_steps=6, config=config)
    print("loss trajectory:", [round(float(l), 4) for l in losses])

    # ---- 6. Greedy next-token pick from the final-step logits ----
    with torch.no_grad():
        final_logits = run_transformer_forward(src_batch, tgt_batch, model_params, num_heads, pad_id)
    next_tok = pick_next_token_by_argmax(final_logits[:, -1, :])
    print("argmax next tokens per row:", next_tok.tolist())
    print("decoded:", [decode_ids_to_tokens([int(t)], id2tok) for t in next_tok])
