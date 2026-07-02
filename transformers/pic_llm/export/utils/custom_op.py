
import torch
import torch.nn.functional as F

class FakeLinearOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, has_bias, name):
        # These become the operator attributes.
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "has_bias_i": has_bias,
            "name_s": name
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FakeLinear", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, has_bias, name):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class FakeLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, has_bias, name):
        super(FakeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.name = name

    def forward(self, x):
        return FakeLinearOp.apply(x, self.in_features, self.out_features, self.has_bias, self.name)

class PicSiluMulOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, gate, up, name):
        kwargs = {
            "name_s": name,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        output_type = gate.type().with_sizes(_get_tensor_sizes(gate))
        return g.op("LlmExporter::PicSiluMul", gate, up, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, gate, up, name):
        return F.silu(gate) * up

class PicSiluMul(torch.nn.Module):
    def __init__(self, name):
        super(PicSiluMul, self).__init__()
        self.name = name

    def forward(self, gate, up):
        return PicSiluMulOp.apply(gate, up, self.name)

class PicGateUpWeightOnlyOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, gate_name, up_name, name):
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "gate_name_s": gate_name,
            "up_name_s": up_name,
            "name_s": name,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        gate, up = g.op("LlmExporter::PicGateUpWeightOnly", input, **kwargs, outputs=2)
        return gate.setType(output_type), up.setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, gate_name, up_name, name):
        out_shape = list(input.shape)[:-1] + [out_features]
        gate = input.new_zeros(out_shape)
        up = input.new_zeros(out_shape)
        return gate, up

class PicGateUpWeightOnly(torch.nn.Module):
    def __init__(self, in_features, out_features, gate_name, up_name, name):
        super(PicGateUpWeightOnly, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_name = gate_name
        self.up_name = up_name
        self.name = name

    def forward(self, x):
        return PicGateUpWeightOnlyOp.apply(
            x, self.in_features, self.out_features, self.gate_name, self.up_name, self.name)

class PicGateUpSiluWeightOnlyOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, gate_name, up_name, name, split_fusion, silu_nhwc_down_fusion):
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "gate_name_s": gate_name,
            "up_name_s": up_name,
            "name_s": name,
            "split_fusion_i": int(split_fusion),
            "silu_nhwc_down_fusion_i": int(silu_nhwc_down_fusion),
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::PicGateUpSiluWeightOnly", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, gate_name, up_name, name, split_fusion, silu_nhwc_down_fusion):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class PicGateUpSiluWeightOnly(torch.nn.Module):
    def __init__(self, in_features, out_features, gate_name, up_name, name, split_fusion=False,
                 silu_nhwc_down_fusion=False):
        super(PicGateUpSiluWeightOnly, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_name = gate_name
        self.up_name = up_name
        self.name = name
        self.split_fusion = bool(split_fusion)
        self.silu_nhwc_down_fusion = bool(silu_nhwc_down_fusion)

    def forward(self, x):
        return PicGateUpSiluWeightOnlyOp.apply(
            x, self.in_features, self.out_features, self.gate_name, self.up_name, self.name,
            self.split_fusion, self.silu_nhwc_down_fusion)

class PicAdrenoGateUpSiluWeightOnlyOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, out_features, gate_name, up_name, name, split_fusion, silu_nhwc_down_fusion):
        kwargs = {
            "in_features_i": in_features,
            "out_features_i": out_features,
            "gate_name_s": gate_name,
            "up_name_s": up_name,
            "name_s": name,
            "split_fusion_i": int(split_fusion),
            "silu_nhwc_down_fusion_i": int(silu_nhwc_down_fusion),
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [out_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::PicAdrenoGateUpSiluWeightOnly", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, out_features, gate_name, up_name, name, split_fusion, silu_nhwc_down_fusion):
        out_shape = list(input.shape)[:-1] + [out_features]
        return input.new_zeros(out_shape)

class PicAdrenoGateUpSiluWeightOnly(torch.nn.Module):
    def __init__(self, in_features, out_features, gate_name, up_name, name, split_fusion=False,
                 silu_nhwc_down_fusion=False):
        super(PicAdrenoGateUpSiluWeightOnly, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_name = gate_name
        self.up_name = up_name
        self.name = name
        self.split_fusion = bool(split_fusion)
        self.silu_nhwc_down_fusion = bool(silu_nhwc_down_fusion)

    def forward(self, x):
        return PicAdrenoGateUpSiluWeightOnlyOp.apply(
            x, self.in_features, self.out_features, self.gate_name, self.up_name, self.name,
            self.split_fusion, self.silu_nhwc_down_fusion)

class PicAdrenoTinyMlpWeightOnlyOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, in_features, inter_features, gate_name, up_name, down_name, name):
        kwargs = {
            "in_features_i": in_features,
            "inter_features_i": inter_features,
            "gate_name_s": gate_name,
            "up_name_s": up_name,
            "down_name_s": down_name,
            "name_s": name,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(input)[:-1] + [in_features]
        output_type = input.type().with_sizes(out_sizes)
        return g.op("LlmExporter::PicAdrenoTinyMlpWeightOnly", input, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, input, in_features, inter_features, gate_name, up_name, down_name, name):
        out_shape = list(input.shape)[:-1] + [in_features]
        return input.new_zeros(out_shape)

class PicAdrenoTinyMlpWeightOnly(torch.nn.Module):
    def __init__(self, in_features, inter_features, gate_name, up_name, down_name, name):
        super(PicAdrenoTinyMlpWeightOnly, self).__init__()
        self.in_features = in_features
        self.inter_features = inter_features
        self.gate_name = gate_name
        self.up_name = up_name
        self.down_name = down_name
        self.name = name

    def forward(self, x):
        return PicAdrenoTinyMlpWeightOnlyOp.apply(
            x, self.in_features, self.inter_features, self.gate_name, self.up_name,
            self.down_name, self.name)

class FusedAttentionOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index, kv_shared_layer_index):
        # These become the operator attributes.
        kwargs = {
            "output_dim_i": output_dim,
            "kv_cache_i": kv_cache,
            "name_s": name,
            "layer_index_i": layer_index,
            "kv_shared_layer_index_i": kv_shared_layer_index,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(query)
        out_sizes[-1] = output_dim
        output_type = query.type().with_sizes(out_sizes)
        return g.op("LlmExporter::FusedAttention", query, key, value, attention_mask, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index, kv_shared_layer_index):
        out_shape = list(query.shape)[:2] + [output_dim]
        return query.new_zeros(out_shape)

class FusedAttention(torch.nn.Module):
    def __init__(self, hidden_size, kv_cache, name, layer_index=-1, kv_shared_layer_index=-1):
        super(FusedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.kv_cache = int(kv_cache)
        self.name = name
        self.layer_index = layer_index
        self.kv_shared_layer_index = kv_shared_layer_index

    def forward(self, query, key, value, attention_mask):
        return FusedAttentionOp.apply(query, key, value, attention_mask, self.hidden_size, self.kv_cache, self.name, self.layer_index, self.kv_shared_layer_index)

class PagedAttentionOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index, kv_shared_layer_index,
                 op_type):
        kwargs = {
            "output_dim_i": output_dim,
            "kv_cache_i": kv_cache,
            "name_s": name,
            "layer_index_i": layer_index,
            "kv_shared_layer_index_i": kv_shared_layer_index,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(query)
        out_sizes[-1] = output_dim
        output_type = query.type().with_sizes(out_sizes)
        return g.op(f"LlmExporter::{op_type}", query, key, value, attention_mask, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, output_dim, kv_cache, name, layer_index, kv_shared_layer_index,
                op_type):
        out_shape = list(query.shape)[:2] + [output_dim]
        return query.new_zeros(out_shape)

class PagedAttentionWithBudgetOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query, key, value, attention_mask, recompute_budget, output_dim, kv_cache, name,
                 layer_index, kv_shared_layer_index, op_type):
        kwargs = {
            "output_dim_i": output_dim,
            "kv_cache_i": kv_cache,
            "name_s": name,
            "layer_index_i": layer_index,
            "kv_shared_layer_index_i": kv_shared_layer_index,
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        query_sizes = _get_tensor_sizes(query)
        out_sizes = list(query_sizes)
        out_sizes[1] = None
        out_sizes[-1] = output_dim
        output_type = query.type().with_sizes(out_sizes)
        index_type = recompute_budget.type().with_sizes([None])
        attn, active_indices = g.op(f"LlmExporter::{op_type}", query, key, value, attention_mask,
                                    recompute_budget, **kwargs, outputs=2)
        return attn.setType(output_type), active_indices.setType(index_type)

    @staticmethod
    def forward(ctx, query, key, value, attention_mask, recompute_budget, output_dim, kv_cache, name,
                layer_index, kv_shared_layer_index, op_type):
        budget_value = int(recompute_budget.reshape(-1)[0].item())
        out_shape = [query.shape[0], budget_value, output_dim]
        active_indices = torch.arange(budget_value, dtype=torch.int32, device=query.device)
        return query.new_zeros(out_shape), active_indices

class PagedAttention(torch.nn.Module):
    def __init__(self, hidden_size, kv_cache, name, layer_index=-1, kv_shared_layer_index=-1):
        super(PagedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.kv_cache = int(kv_cache)
        self.name = name
        self.layer_index = layer_index
        self.kv_shared_layer_index = kv_shared_layer_index

    def forward(self, query, key, value, attention_mask, recompute_budget=None, op_type='PagedAttention'):
        if recompute_budget is not None:
            return PagedAttentionWithBudgetOp.apply(query, key, value, attention_mask, recompute_budget,
                                                    self.hidden_size, self.kv_cache, self.name, self.layer_index,
                                                    self.kv_shared_layer_index, op_type)
        return PagedAttentionOp.apply(query, key, value, attention_mask, self.hidden_size, self.kv_cache, self.name,
                                      self.layer_index, self.kv_shared_layer_index, op_type)

class MoEOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, hidden_states, routing_weights, selected_experts, num_experts, top_k, layer_id):
        kwargs = {
            "num_experts_i": num_experts,
            "top_k_i": top_k,
            "layer_id_i": layer_id
        }
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        out_sizes = _get_tensor_sizes(hidden_states)
        output_type = hidden_states.type().with_sizes(out_sizes)
        return g.op("LlmExporter::MoE", hidden_states, routing_weights, selected_experts, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, hidden_states, routing_weights, selected_experts, num_experts, top_k, layer_id):
        return hidden_states

class MoE(torch.nn.Module):
    def __init__(self, num_experts, top_k, layer_id):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_id = layer_id

    def forward(self, hidden_states, routing_weights, selected_experts):
        return MoEOp.apply(hidden_states, routing_weights, selected_experts, self.num_experts, self.top_k, self.layer_id)

class FusedLinearAttentionOp(torch.autograd.Function):
    """
    Unified Custom Op for Linear Attention variants.

    Inputs (Tensors):
      0: qkv          [B, D, L]   - QKV projection output (before conv)
      1: gate         [B, L, H]   - Pre-computed decay factor
      2: beta         [B, L, H]   - Pre-computed learning rate (optional)
      3: conv_weight  [D, 1, K]   - Causal conv weight (optional)
      4: conv_bias    [D]         - Causal conv bias (optional)

    Attributes:
      type:             "gated_delta_rule" | "mamba" | "rwkv" | "gla" | "retnet"
      key_dim:          K total dim = num_k_heads * head_k_dim
      value_dim:        V total dim = num_v_heads * head_v_dim
      num_k_heads:      number of K/Q heads
      num_v_heads:      number of V heads (may differ from K for GQA)
      head_k_dim:       per-head K dimension
      head_v_dim:       per-head V dimension
      use_qk_l2norm:    whether to L2-normalize Q and K

    Output:
      0: attn_out  [B, L, num_v_heads, head_v_dim]

    Internal State (managed by MNN Execution, not in graph):
      conv_state:  [B, D, kernel_size - 1]
      rnn_state:   [B, num_v_heads, head_k_dim, head_v_dim]
    """
    @staticmethod
    def symbolic(g, qkv, gate, beta, conv_weight, name, attn_type,
                 num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        kwargs = {
            "name_s": name,
            "attn_type_s": attn_type,
            "num_k_heads_i": num_k_heads,
            "num_v_heads_i": num_v_heads,
            "head_k_dim_i": head_k_dim,
            "head_v_dim_i": head_v_dim,
            "use_qk_l2norm_i": int(use_qk_l2norm)
        }
        inputs = [qkv, gate, beta, conv_weight]
        from torch.onnx.symbolic_helper import _get_tensor_sizes
        qkv_sizes = _get_tensor_sizes(qkv)
        # qkv shape is [Batch, Dim, SeqLen]
        batch_size = qkv_sizes[0]
        seq_len = qkv_sizes[2]

        out_sizes = [batch_size, seq_len, num_v_heads, head_v_dim]
        output_type = qkv.type().with_sizes(out_sizes)

        return g.op("LlmExporter::FusedLinearAttention", *inputs, **kwargs).setType(output_type)

    @staticmethod
    def forward(ctx, qkv, gate, beta, conv_weight, name, attn_type,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        # Dummy forward: return correct output shape
        # qkv: [B, D, L] -> output: [B, L, num_v_heads, head_v_dim]
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[2]
        return qkv.new_zeros([batch_size, seq_len, num_v_heads, head_v_dim])

class FusedLinearAttention(torch.nn.Module):
    def __init__(self, name, attn_type, num_k_heads, num_v_heads, head_k_dim, head_v_dim, use_qk_l2norm):
        super(FusedLinearAttention, self).__init__()
        self.name = name
        self.attn_type = attn_type
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, qkv, gate, beta, conv_weight):
        return FusedLinearAttentionOp.apply(qkv, gate, beta, conv_weight, self.name,
                                      self.attn_type, self.num_k_heads, self.num_v_heads,
                                      self.head_k_dim, self.head_v_dim, self.use_qk_l2norm)
