import torch

from ..modules.linear import FP8Linear, Q8Linear


def get_device_arch():
    major, minor = torch.cuda.get_device_capability(0)
    if major == 8 and (minor >= 0 and minor < 9):
        return "ampere"
    if major == 8 and minor == 9:
        return "ada"
    if major == 9 and minor == 0:
        return "hopper"
    if major == 12:
        return "blackwell"
    raise NotImplementedError("Not supported gpu!")


def is_fa3_available():
    try:
        import flash_attn_interface

        return True
    except ImportError:
        return False


def is_fp8_attention_available():
    try:
        from q8_kernels_cuda.flash_attention._C import flash_attention_fp8

        return True
    except ImportError:
        return False


def get_attention_func(use_fp8_attention=False):
    try:
        device_arch = get_device_arch()
        print(f"🔍 Debug: device_arch = {device_arch}, use_fp8_attention = {use_fp8_attention}")
        use_default = False
        
        # Initialize variables to ensure they're always defined
        self_attn_func = None
        self_attn_memory_layout = None
        out_tuple = False
        
        if device_arch == "hopper":
            if use_fp8_attention:
                assert (
                    is_fa3_available()
                ), "FP8 attention is not available! Install flash-attn-3 first."
            if is_fa3_available():
                from flash_attn_interface import flash_attn_func

                @torch.library.custom_op("fa3::attn_func", mutates_args=())
                def _fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return flash_attn_func(q, k, v, causal=False)[0]

                @_fn.register_fake
                def _fn_fake(
                    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    return torch.empty_like(q, dtype=torch.bfloat16)

                self_attn_func = _fn
                self_attn_memory_layout = lambda x: x
                out_tuple = False
                print("🔍 Debug: Using hopper flash attention")
            else:
                use_default = True
                print("🔍 Debug: hopper flash attention not available, using default")
        elif device_arch == "ada" or device_arch == "blackwell":
            if use_fp8_attention:
                assert (
                    is_fp8_attention_available()
                ), "FP8 attention is not available! Use Ada or Blackwell GPU."
            if use_fp8_attention and is_fp8_attention_available():
                from q8_kernels_cuda.flash_attention._C import flash_attention_fp8

                @torch.library.custom_op("fa_fp8::attn_func", mutates_args=())
                def _fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    v_fp8 = v = v.transpose(-2, -1).contiguous()  # b h s d -> b h d s
                    if v_fp8.shape[-1] % 16 != 0:
                        v_tokens = v_fp8.shape[-1]
                        v_tokens_pad = ((v_tokens + 15) // 16) * 16 - v_tokens
                        v_fp8 = torch.nn.functional.pad(v_fp8, (0, v_tokens_pad))
                    softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
                    return flash_attention_fp8(
                        q, k, v_fp8.to(torch.float8_e4m3fn), softmax_scale, True
                    )

                @_fn.register_fake
                def _fn_fake(
                    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    return torch.empty_like(q, dtype=torch.bfloat16)

                self_attn_func = _fn
                self_attn_memory_layout = lambda x: x.transpose(1, 2).contiguous()
                out_tuple = False
                print("🔍 Debug: Using ada/blackwell FP8 attention")
            else:
                use_default = True
                print("🔍 Debug: ada/blackwell FP8 attention not available, using default")
        else:
            # For any other architecture (like ampere), use default
            use_default = True
            print(f"🔍 Debug: Unknown architecture {device_arch}, using default")

        if use_default or self_attn_func is None:
            from flash_attn import flash_attn_func

            @torch.library.custom_op("fa::attn_func", mutates_args=())
            def _fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return flash_attn_func(q, k, v, causal=False)[0]

            @_fn.register_fake
            def _fn_fake(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return torch.empty_like(q, dtype=torch.bfloat16)

            self_attn_func = _fn
            self_attn_memory_layout = lambda x: x.transpose(1, 2).contiguous()
            out_tuple = False
            print("🔍 Debug: Using default flash attention")

        print(f"🔍 Debug: Returning self_attn_func = {self_attn_func}")
        return self_attn_func, self_attn_memory_layout, out_tuple
    
    except Exception as e:
        print(f"❌ Error in get_attention_func: {e}")
        # Fallback to basic attention
        self_attn_func = torch.nn.functional.scaled_dot_product_attention
        self_attn_memory_layout = lambda x: x.transpose(1, 2).contiguous()
        out_tuple = False
        print("🔍 Debug: Using fallback scaled_dot_product_attention")
        return self_attn_func, self_attn_memory_layout, out_tuple
        self_attn_func = torch.nn.functional.scaled_dot_product_attention
        self_attn_memory_layout = lambda x: x.transpose(1, 2)
        out_tuple = False

    cross_attn_func = torch.nn.functional.scaled_dot_product_attention
    cross_attn_memory_layout = lambda x: x.transpose(1, 2)

    return (
        (self_attn_func, self_attn_memory_layout),
        (cross_attn_func, cross_attn_memory_layout),
        out_tuple,
    )


def get_compute_dtype():
    arch = get_device_arch()
    if arch == "hopper" or arch == "ada" or arch == "blackwell":
        return torch.float8_e4m3fn
    else:
        return torch.int8


def get_linear_cls():
    arch = get_device_arch()
    if arch == "hopper" or arch == "ada" or arch == "blackwell":
        return FP8Linear
    else:
        return Q8Linear
