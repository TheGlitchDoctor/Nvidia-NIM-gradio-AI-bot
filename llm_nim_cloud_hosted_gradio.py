# Using a LLM NIM that is running on Nvidia's servers to respond to a query related to Module scripts
from typing import Iterator, Optional, Tuple, List
import openai
import os
from dotenv import load_dotenv
import gradio as gr
from faster_whisper import WhisperModel
import tempfile

RUN_WITH_UI = True

# Load environment variables once at startup
load_dotenv()

# Global variables for API configuration
AVAILABLE_MODELS = {
    "llama-3.1-8b": "NVIDIA_LLAMA_3pt1_8B_INSTRUCT_MODEL",
    "llama-3.1-405b": "NVIDIA_LLAMA_3pt1_405B_INSTRUCT_MODEL",
    "deepseek": "NVIDIA_DEEPSEEK_R1_MODEL"
}

def get_module_examples() -> str:
    """
    Returns example scripts for using the ModuleName library.
    Reads examples from the module_examples.py file.
    """
    from module_examples import EXAMPLES
    return EXAMPLES

def get_api_config():
    """Get API configuration from environment variables with proper error handling."""
    model_name = os.getenv("NVIDIA_LLAMA_3pt1_405B_INSTRUCT_MODEL")
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it in your .env file.")
    
    if not model_name:
        raise ValueError("Model environment variable is not set. Please set it in your .env file.")
    
    return model_name, api_key

def create_system_context() -> str:
    """
    Create the initial system context with module examples.
    """
    # Base context
    context = "You are an expert in Python scripting for the {module_name} library."
    
    # Add examples
    examples = get_module_examples()
    if examples:
        context += "\n\nHere are examples of using the {module_name} library:\n\n" + examples
    
    return context

def get_completion_response(model_name: str, api_key: str, messages: list) -> Iterator[str]:
    """
    Get streaming completion response from the API.
    
    Args:
        model_name: Name of the model to use
        api_key: API key for authentication
        messages: List of conversation messages
    
    Yields:
        Tokens from the API response
    """
    try:
        client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        error_message = f"API call failed: {str(e)}"
        yield error_message
        raise

def ask_llm(prompt: str, history: Optional[List[dict]] = None) -> Iterator[Tuple[List[dict], str]]:
    """
    Process a user query using the LLM.
    
    Args:
        prompt: User's input query
        history: Conversation history (default: None)
    
    Yields:
        Updated history and empty string for textbox
    """
    if not prompt or prompt.isspace():
        yield history or [], ""
        return

    # Initialize history if None
    history = history or []
    
    # Convert history to messages format if it's not already
    messages = []
    for msg in history:
        if isinstance(msg, list):
            # Convert old format [user_msg, assistant_msg] to new format
            messages.extend([
                {"role": "user", "content": msg[0]},
                {"role": "assistant", "content": msg[1]}
            ])
        else:
            # Already in correct format
            messages.append(msg)
    
    # Add system context and user prompt
    api_messages = [
        {"role": "system", "content": SYSTEM_CONTEXT},
        *messages,
        {"role": "user", "content": prompt}
    ]

    # Get API config
    model_name, api_key = get_api_config()

    try:
        # Get streaming response
        response = get_completion_response(model_name, api_key, api_messages)
        
        # Process each token
        full_response = ""
        for token in response:
            full_response += token
            
            # Update history with current response
            current_history = messages + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_response}
            ]
            
            yield current_history, ""
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        current_history = messages + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": error_msg}
        ]
        yield current_history, ""

# Initialize system context after all functions are defined
SYSTEM_CONTEXT = create_system_context()

# Initialize Whisper model (small model for faster processing)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    """
    Transcribe audio using faster-whisper model.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    import subprocess
    
    try:
        # Define the command to call the external script
        command = [
            "python", "./python-clients/scripts/asr/transcribe_file.py",
            "--server", "grpc.nvcf.nvidia.com:443",
            "--use-ssl",
            "--metadata", "function-id", "1598d209-5e27-4d3c-8079-4751568b1081",
            "--metadata", "authorization", f"Bearer {os.getenv('NVIDIA_API_KEY')}",  # Use API key
            "--language-code", "en-US",
            "--input-file", audio_path
        ]
        
        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            print(result.stdout.strip())
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Error in transcription: {result.stderr.strip()}")
    
    except Exception as e:
        return f"Error: {str(e)}"
    
    # segments, _ = whisper_model.transcribe(audio_path)
    # return " ".join([segment.text for segment in segments])

def process_audio(audio, chatbot):
    """
    Process recorded audio and add transcription to chat.
    
    Args:
        audio: Audio data from Gradio
        chatbot: Current chat state
        
    Returns:
        Updated chat state and transcribed text
    """
    if audio is None:
        return chatbot, ""  # Return empty string without triggering chat
        
    # Transcribe the audio
    text = transcribe_audio(audio)
    
    if not text or text.isspace():  # Check if transcription is empty or just whitespace
        return chatbot, ""  # Return empty string without triggering chat
        
    return chatbot, text

def handle_empty_input(text, chatbot):
    """Helper function to handle empty input"""
    if not text or text.isspace():
        return chatbot, ""
    return chatbot, text  # Return the input to be processed by ask_llm

def handle_example(text, history):
    response = next(ask_llm(text, history))
    return response[0], ""

# Set up the UI
if __name__ == "__main__":
    if not RUN_WITH_UI:
        # No UI, just run using the following prompt and print to terminal
        for response in ask_llm(prompt="Can you write me a module script that creates a demo", history=[]):
            print(response[0][-1]["content"])  # Print just the assistant's response
    else:
        # Create a custom Gradio interface with logo
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            # Header section with logo and title
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Image(
                        "images/ai_agent.jpg",
                        show_label=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                        height=80,
                        interactive=False,
                        container=False
                    )
                with gr.Column(scale=4):
                    gr.Markdown("# Module_name Code Assistant")
                    gr.Markdown("I can help you with module_name scripts and answer questions.")
            
            # Chat interface
            with gr.Row():
                with gr.Column(scale=4):
                    # Add example buttons at the top
                    with gr.Column(visible=True, elem_classes="examples-container"):
                        gr.Markdown("Here are some example questions you can ask:")
                        with gr.Row():
                            example1 = gr.Button("Create a module_name script", size="sm")
                            example2 = gr.Button("How can I create a demo script?", size="sm")
                    
                    # Create textbox for storing example text
                    example_text = gr.Textbox(visible=False)
                    
                    # Add chatbot below examples
                    chatbot = gr.Chatbot(
                        height=620,  # Reduced to fit screen better
                        container=False,
                        show_label=False,
                        type="messages"  # Use messages type instead of tuples
                    )
                    
                    # Chat input with send button
                    with gr.Row():
                        txt = gr.Textbox(
                            placeholder="Ask me about Module_name...",
                            container=False,
                            show_label=False,
                            scale=20
                        )
                        send_btn = gr.Button("⮞", scale=1, min_width=40)
                    
                    def set_example_text(text):
                        return text
                    
                    # Connect example buttons
                    for btn, text in [
                        (example1, "Create a demo script."),
                        (example2, "How can I create a demo script?")
                    ]:
                        btn.click(
                            fn=set_example_text,
                            inputs=[gr.Textbox(value=text, visible=False)],
                            outputs=[txt]
                        ).then(
                            fn=ask_llm,
                            inputs=[txt, chatbot],
                            outputs=[chatbot, txt]
                        ).then(
                            lambda: None,
                            None,
                            [txt],
                            queue=False
                        )
                    
                with gr.Column(scale=1, min_width=150):
                    # Add empty space to match examples section height
                    gr.Markdown("&nbsp;", elem_classes="voice-input-spacer")
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Voice Input"
                    )
            
            # Add custom CSS for the example buttons and voice input
            gr.Markdown("""
                <style>
                .examples-container {
                    border: none !important;
                    background: #1F1F1F !important;
                    margin-bottom: 1em !important;
                    padding: 1em !important;  
                    border-radius: 8px !important;
                }
                .gr-button.gr-button-sm {
                    margin: 0.2em !important;
                    background: #2B2B2B !important;
                    border: none !important;
                    color: #fff !important;
                    font-size: 2.3em !important;
                    padding: 0.5em 1em !important;
                }
                .gr-button.gr-button-sm:hover {
                    background: #3B3B3B !important;
                    transform: translateY(-1px);
                }
                .voice-input-spacer {
                    margin-top: 7em !important;  /* Reduced to match shorter chat window */
                }
                </style>
            """)
            
            # Handle voice input and automatically submit
            audio_input.change(
                fn=process_audio,
                inputs=[audio_input, chatbot],
                outputs=[chatbot, txt]
            ).then(  # Chain the ask_llm function to run only if there's text
                fn=handle_empty_input,
                inputs=[txt, chatbot],
                outputs=[chatbot, txt]
            ).then(  # Now process with ask_llm if we have text
                fn=ask_llm,
                inputs=[txt, chatbot],
                outputs=[chatbot, txt]
            ).then(  # Clear the text input
                lambda: None,
                None,
                [txt],
                queue=False
            )
            
            # Chat submit for text input
            txt.submit(
                fn=ask_llm,
                inputs=[txt, chatbot],
                outputs=[chatbot, txt]
            ).then(
                lambda: None,
                None,
                [txt],
                queue=False
            )

            send_btn.click(
                fn=ask_llm,
                inputs=[txt, chatbot],
                outputs=[chatbot, txt]
            ).then(
                lambda: None,
                None,
                [txt],
                queue=False
            )
        
        # Launch with a larger default size
        demo.launch(height=800, width=1000, share=False, server_name="0.0.0.0")