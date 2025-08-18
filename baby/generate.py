# Model parallel inference
# Note: This script is for demonstration purposes only. It is not designed for production use.
#       See gpt_oss.chat for a more complete example with the Harmony parser.
# torchrun --nproc-per-node=4 -m gpt_oss.generate -p "why did the chicken cross the road?" model/

import argparse

from gpt_oss.tokenizer import get_tokenizer


def main(args):
    match args.backend:
        case "torch":
            from gpt_oss.torch.utils import init_distributed
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device=device)
        case "triton":
            from gpt_oss.torch.utils import init_distributed
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, context=args.context_length, device=device)
        case "vllm":
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=args.tensor_parallel_size)
        case "gyro":
            # GyroSI backend - use responses API inference wrapper
            from baby.responses_api.inference.gyro import setup_model
            from openai_harmony import load_harmony_encoding, HarmonyEncodingName
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            infer_next_token = setup_model(
                checkpoint=args.checkpoint,
                encoding=encoding,
                config_path=getattr(args, 'config', 'baby/config.json')
            )
            # Create a simple generator wrapper for GyroSI
            class GyroGenerator:
                def __init__(self, infer_fn, encoding):
                    self.infer_fn = infer_fn
                    self.encoding = encoding
                    
                def generate(self, tokens, **kwargs):
                    current_tokens = list(tokens)
                    max_tokens = kwargs.get('max_tokens', 1000)
                    stop_tokens = kwargs.get('stop_tokens', [])
                    temperature = kwargs.get('temperature', 0.0)
                    return_logprobs = kwargs.get('return_logprobs', False)
                    
                    for _ in range(max_tokens):
                        next_token = self.infer_fn(current_tokens, temperature)
                        if next_token in stop_tokens:
                            break
                        current_tokens.append(next_token)
                        if return_logprobs:
                            yield next_token, 0.0  # GyroSI is deterministic
                        else:
                            yield next_token
                        
            generator = GyroGenerator(infer_next_token, encoding)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(args.prompt)
    max_tokens = None if args.limit == 0 else args.limit
    for token, logprob in generator.generate(tokens, stop_tokens=[tokenizer.eot_token], temperature=args.temperature, max_tokens=max_tokens, return_logprobs=True):
        tokens.append(token)
        token_text = tokenizer.decode([token])
        print(
            f"Generated token: {repr(token_text)}, logprob: {logprob}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation example")
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="How are you?",
        help="LLM prompt",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "-l",
        "--limit",
        metavar="LIMIT",
        type=int,
        default=0,
        help="Limit on the number of tokens (0 to disable)",
    )
    parser.add_argument(
        "-b",
        "--backend",
        metavar="BACKEND",
        type=str,
        default="torch",
        choices=["triton", "torch", "vllm", "gyro"],
        help="Inference backend",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        help="Path to the GyroSI configuration file (for gyro backend)",
        default="baby/config.json",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM backend",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length for Triton backend",
    )
    args = parser.parse_args()

    main(args)
