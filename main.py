# Import Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# Load the Model and Tokenizer
model_name_or_path = "TheBloke/meditron-70B-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device="cuda"  # Change the device to "cpu"
)

# Sample system message
system_message = "This is a system message."

prompt = "What is the role of AI in managing cardiovascular disease?"
prompt_template = f'''system
{system_message}
user
{prompt}
assistant
'''

# Convert prompt to tokens
tokens = tokenizer.encode(prompt_template, return_tensors='pt')

generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}

# Generation with a streamer
streamer = TextGenerationPipeline(model=model, tokenizer=tokenizer)
generation_output_streamed = streamer(
    prompt_template,
    skip_special_tokens=True,
    **generation_params
)

# Get the tokens from the output, decode them, and print them
token_output_streamed = generation_output_streamed[0]['generated_token_ids']
text_output_streamed = tokenizer.decode(token_output_streamed, skip_special_tokens=True)
print("model.generate(streamed) output: ", text_output_streamed)

# Inference using Transformer's pipeline
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
pipe_output = pipe(prompt_template, **generation_params)[0]['generated_text']
print(pipe_output)