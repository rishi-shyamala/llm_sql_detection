1. Download base file from huggingface (only need to do this once)
    - `hf download unsloth/Qwen3-4B-Instruct-2507`
2. Convert original file to GGUF 
    - `python llama.cpp/convert_hf_to_gguf.py  --outfile ./Qwen3-4B-Instruct.gguf Qwen3-4B-Instruct/`
3. Convert lora to gguf 
    - `python ../llama.cpp/convert_lora_to_gguf.py --base ./Qwen3-4B-Instruct --outfile ./lora_adaptor.gguf --outtype f16 ../outputs/run_20251107-144353/grpo_saved_lora`
4. merge lora and original gguf into one file
    - `llama-export-lora -m Qwen3-4B-Instruct.gguf --lora lora_adaptor.gguf -o Qwen3-4B-merged-F16.gguf`
5. Quantize large gguf into smaller model
    - `llama-quantize Qwen3-4B-merged-F16.gguf Qwen3-4B-merged-Q4_K_M.gguf Q4_K_M`
6. Create Modelfile for Ollama
    
```
FROM ./Qwen3-4B-merged-Q4_K_M.gguf
SYSTEM "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"
TEMPLATE """
{{- if .Messages }}
  {{- range .Messages }}
{{ .Content }}

  {{- end }}
{{- else }}
You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>.

{{ .Prompt }}

{{- end }}
<start_working_out>
"""
PARAMETER temperature 0.7
PARAMETER min_p 0.0
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER stop "<|im_end|>"
```
7. Load into Ollama 
    - `ollama create qwen3-finetuned -f Modelfile`