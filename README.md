# Module Name LLM Demo

This repository contains a demonstration of using NVIDIA's Large Language Models (LLMs) to assist with Module Name scripting tasks. The demo features a Gradio-based web interface that allows users to interact with the LLM and get assistance with Module Name scripts.

## Overview

This application combines:
- NVIDIA's hosted LLM service
- Module Name library knowledge
- Gradio web interface for easy interaction
- Voice input capabilities using Whisper

The LLM is trained to understand Module Name scripting patterns and can help users create scripts for various module tasks.

## Prerequisites

- Python 3.x
- Required Python packages:
  - openai>=1.0.0
  - gradio>=4.0.0
  - python-dotenv>=1.0.0
  - faster-whisper>=0.10.0

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:
```
NVIDIA_LLAMA_3pt1_405B_INSTRUCT_MODEL=your_model_name
NVIDIA_API_KEY=your_api_key
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python llm_nim_cloud_hosted_module_gradio.py
```

2. Once running, the Gradio interface will launch in your default web browser
3. Enter your Module Name-related questions or requests in the interface
4. Use the example buttons for quick access to common queries
5. Optionally use voice input for hands-free interaction
6. The LLM will provide responses and code examples based on your queries

## Features

- Interactive chat interface with streaming responses
- Context-aware conversations with chat history
- Voice input support using Whisper
- Pre-configured example questions
- Real-time response streaming
- Specialized knowledge of Module Name scripting

## Example Questions

The interface includes example questions to help you get started:
1. "Create a demo script"
2. "How can I create a demo script?"

These examples demonstrate common Module Name tasks and can be clicked to instantly get detailed responses.
