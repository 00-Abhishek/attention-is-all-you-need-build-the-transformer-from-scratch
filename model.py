"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    token_to_id={}


    #add special tokens first
    for token in specials:
        if token not in token_to_id:
            token_to_id[token]=len(token_to_id)

    #for corpus tokens in first-seen order

    for sentence in sentences:
        for token in sentence.split():
            if token not in token_to_id:
                token_to_id[token]=len(token_to_id)
    return token_to_id
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.

# Step 2 - build_id_to_token_vocab (not yet solved)
# TODO: implement

# Step 3 - encode_sentence_to_ids (not yet solved)
# TODO: implement

# Step 4 - decode_ids_to_tokens (not yet solved)
# TODO: implement

# Step 5 - pad_id_sequence (not yet solved)
# TODO: implement

# Step 6 - stack_padded_sequences_to_batch (not yet solved)
# TODO: implement

# Step 7 - scale_embeddings_by_sqrt_d_model (not yet solved)
# TODO: implement

# Step 8 - compute_positional_div_term (not yet solved)
# TODO: implement

# Step 9 - build_position_index_column (not yet solved)
# TODO: implement

# Step 10 - fill_even_indices_with_sin (not yet solved)
# TODO: implement

# Step 11 - fill_odd_indices_with_cos (not yet solved)
# TODO: implement

# Step 12 - build_sinusoidal_positional_encoding (not yet solved)
# TODO: implement

# Step 13 - add_positional_encoding_to_embeddings (not yet solved)
# TODO: implement

# Step 14 - build_padding_mask (not yet solved)
# TODO: implement

# Step 15 - build_causal_mask (not yet solved)
# TODO: implement

# Step 16 - combine_padding_and_causal_masks (not yet solved)
# TODO: implement

# Step 17 - compute_raw_attention_scores (not yet solved)
# TODO: implement

# Step 18 - scale_attention_scores (not yet solved)
# TODO: implement

# Step 19 - mask_attention_scores_with_neg_inf (not yet solved)
# TODO: implement

# Step 20 - softmax_attention_weights (not yet solved)
# TODO: implement

# Step 21 - apply_attention_weights_to_values (not yet solved)
# TODO: implement

# Step 22 - scaled_dot_product_attention (not yet solved)
# TODO: implement

# Step 23 - split_last_dim_into_heads (not yet solved)
# TODO: implement

# Step 24 - transpose_heads_before_sequence (not yet solved)
# TODO: implement

# Step 25 - merge_heads_back_to_model_dim (not yet solved)
# TODO: implement

# Step 26 - apply_linear_projection (not yet solved)
# TODO: implement

# Step 27 - project_to_query_key_value (not yet solved)
# TODO: implement

# Step 28 - split_qkv_into_heads (not yet solved)
# TODO: implement

# Step 29 - multi_head_scaled_dot_product_attention (not yet solved)
# TODO: implement

# Step 30 - merge_heads_and_project_output (not yet solved)
# TODO: implement

# Step 31 - assemble_multi_head_attention_forward (not yet solved)
# TODO: implement

# Step 32 - apply_ffn_first_linear_and_relu (not yet solved)
# TODO: implement

# Step 33 - apply_ffn_second_linear (not yet solved)
# TODO: implement

# Step 34 - position_wise_feed_forward_network (not yet solved)
# TODO: implement

# Step 35 - compute_layer_norm_mean_and_variance (not yet solved)
# TODO: implement

# Step 36 - normalize_and_scale_with_gamma_beta (not yet solved)
# TODO: implement

# Step 37 - apply_residual_add_and_norm (not yet solved)
# TODO: implement

# Step 38 - apply_dropout_with_keep_mask (not yet solved)
# TODO: implement

# Step 39 - encoder_layer_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 40 - encoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 41 - assemble_encoder_layer (not yet solved)
# TODO: implement

# Step 42 - stack_encoder_layers (not yet solved)
# TODO: implement

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

