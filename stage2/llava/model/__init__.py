# try:
#     from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
#     from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
#     from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
# except:
#     pass
# filepath: [__init__.py](http://_vscodecontentref_/1)

# Option 1: handle each import independently for explicit errors.
try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
except ImportError as e:
    print(f"Warning: Failed to import LlavaLlamaForCausalLM: {e}")

try:
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
except ImportError as e:
    print(f"Warning: Failed to import LlavaMptForCausalLM: {e}")

try:
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except ImportError as e:
    print(f"Warning: Failed to import LlavaMistralForCausalLM: {e}")