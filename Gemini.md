# Gemini API Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [SDK Usage](#sdk-usage)
4. [Configuration Options](#configuration-options)
5. [Basic Text Generation](#basic-text-generation)
6. [Chat Conversations](#chat-conversations)
7. [Vision Capabilities](#vision-capabilities)
8. [Structured Output](#structured-output)
9. [Batch Processing](#batch-processing)
10. [Function Calling](#function-calling)
11. [Safety Settings](#safety-settings)
12. [Token Counting](#token-counting)
13. [Streaming Responses](#streaming-responses)
14. [Error Handling](#error-handling)
15. [Best Practices](#best-practices)
16. [Advanced Features](#advanced-features)

## Introduction

Google's Gemini API is a powerful generative AI platform that provides access to state-of-the-art language models. It supports text generation, chat conversations, vision understanding, function calling, and structured output generation.

### Available Models
- **gemini-1.5-pro**: Most capable model with 2M context window
- **gemini-1.5-flash**: Faster, cost-effective model with 1M context window
- **gemini-1.0-pro**: Legacy model for basic text generation

## Setup and Installation

### 1. Install the SDK

```bash
pip install google-generativeai
```

### 2. Get API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable

```python
import os
import google.generativeai as genai

# Set your API key
os.environ['GOOGLE_API_KEY'] = 'your-api-key-here'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
```

## SDK Usage

### Basic Import and Setup

```python
import google.generativeai as genai
import os
from typing import List, Dict, Any
import json
import time

# Configure the API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# List available models
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model: {model.name}")
```

## Configuration Options

### Model Configuration

```python
# Generation configuration
generation_config = genai.types.GenerationConfig(
    candidate_count=1,
    stop_sequences=['x'],
    max_output_tokens=1000,
    temperature=0.7,
    top_p=0.8,
    top_k=40
)

# Safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Initialize model with configuration
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    generation_config=generation_config,
    safety_settings=safety_settings
)
```

## Basic Text Generation

### Simple Text Generation

```python
def generate_text(prompt: str, model_name: str = 'gemini-1.5-pro') -> str:
    """Generate text using Gemini API"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
prompt = "Write a short story about a robot learning to paint"
result = generate_text(prompt)
print(result)
```

### Text Generation with Custom Parameters

```python
def generate_with_params(prompt: str, temperature: float = 0.7, max_tokens: int = 1000):
    """Generate text with custom parameters"""
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=0.9,
        top_k=40
    )
    
    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        generation_config=generation_config
    )
    
    response = model.generate_content(prompt)
    return response.text

# Example with different creativity levels
creative_response = generate_with_params(
    "Write a creative poem about AI", 
    temperature=0.9, 
    max_tokens=500
)

factual_response = generate_with_params(
    "Explain quantum computing", 
    temperature=0.1, 
    max_tokens=800
)
```

## Chat Conversations

### Basic Chat Implementation

```python
class GeminiChat:
    def __init__(self, model_name: str = 'gemini-1.5-pro'):
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
    
    def send_message(self, message: str) -> str:
        """Send a message and get response"""
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_history(self) -> List[Dict]:
        """Get chat history"""
        return [{"role": msg.role, "content": msg.parts[0].text} 
                for msg in self.chat.history]
    
    def clear_history(self):
        """Clear chat history"""
        self.chat = self.model.start_chat(history=[])

# Example usage
chat = GeminiChat()

# Multi-turn conversation
print(chat.send_message("Hello! Can you help me with Python programming?"))
print(chat.send_message("What are list comprehensions?"))
print(chat.send_message("Can you give me an example?"))

# View conversation history
history = chat.get_history()
for turn in history:
    print(f"{turn['role']}: {turn['content'][:100]}...")
```

### Chat with System Instructions

```python
def create_specialized_chat(system_instruction: str, model_name: str = 'gemini-1.5-pro'):
    """Create a chat with system instructions"""
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction
    )
    return model.start_chat()

# Example: Python tutor
python_tutor = create_specialized_chat(
    "You are an expert Python tutor. Explain concepts clearly with examples. "
    "Always provide working code examples and explain the logic step by step."
)

response = python_tutor.send_message("How do decorators work in Python?")
print(response.text)
```

## Vision Capabilities

### Image Analysis

```python
import PIL.Image

def analyze_image(image_path: str, prompt: str = "Describe this image"):
    """Analyze an image with Gemini Vision"""
    try:
        # Load and prepare image
        image = PIL.Image.open(image_path)
        
        # Use vision model
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content([prompt, image])
        
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Example usage
image_description = analyze_image(
    "path/to/your/image.jpg",
    "What objects do you see in this image? Describe their colors and positions."
)
print(image_description)
```

### Multiple Images Analysis

```python
def analyze_multiple_images(image_paths: List[str], prompt: str):
    """Analyze multiple images together"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Load all images
        images = [PIL.Image.open(path) for path in image_paths]
        
        # Create content with prompt and images
        content = [prompt] + images
        
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Compare images
comparison = analyze_multiple_images(
    ["image1.jpg", "image2.jpg"],
    "Compare these two images. What are the similarities and differences?"
)
```

### Image + Text Analysis

```python
def image_text_analysis(image_path: str, text_context: str, question: str):
    """Analyze image with additional text context"""
    image = PIL.Image.open(image_path)
    
    prompt = f"""
    Context: {text_context}
    
    Question: {question}
    
    Please analyze the image in the context of the provided information.
    """
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([prompt, image])
    
    return response.text

# Example: Document analysis
result = image_text_analysis(
    "chart.png",
    "This is a sales report for Q3 2024",
    "What trends do you see in the data and what recommendations would you make?"
)
```

## Structured Output

### JSON Schema Output

```python
import json
from typing import Dict, List

def generate_structured_output(prompt: str, schema: Dict) -> Dict:
    """Generate structured JSON output"""
    
    structured_prompt = f"""
    {prompt}
    
    Please respond with a valid JSON object that follows this schema:
    {json.dumps(schema, indent=2)}
    
    Important: Return only the JSON object, no additional text.
    """
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(structured_prompt)
    
    try:
        return json.loads(response.text.strip())
    except json.JSONDecodeError:
        # Extract JSON from response if wrapped in markdown
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:-3]
        elif text.startswith('```'):
            text = text[3:-3]
        return json.loads(text)

# Example schema for product analysis
product_schema = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string"},
        "category": {"type": "string"},
        "price_range": {"type": "string"},
        "features": {
            "type": "array",
            "items": {"type": "string"}
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"}
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "rating": {"type": "number", "minimum": 1, "maximum": 10}
    },
    "required": ["product_name", "category", "features", "rating"]
}

# Generate structured product analysis
product_analysis = generate_structured_output(
    "Analyze the iPhone 15 Pro and provide a comprehensive review",
    product_schema
)

print(json.dumps(product_analysis, indent=2))
```

### Data Extraction from Text

```python
def extract_entities(text: str) -> Dict:
    """Extract entities from unstructured text"""
    
    extraction_schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "organization": {"type": "string"}
                    }
                }
            },
            "organizations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "dates": {
                "type": "array",
                "items": {"type": "string"}
            },
            "key_topics": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    prompt = f"""
    Extract entities from the following text and structure them according to the schema:
    
    Text: {text}
    """
    
    return generate_structured_output(prompt, extraction_schema)

# Example usage
news_text = """
    Apple CEO Tim Cook announced yesterday that the company will invest $1 billion 
    in AI research over the next two years. The announcement was made at the 
    company's headquarters in Cupertino, California, during a meeting with investors.
    """

entities = extract_entities(news_text)
print(json.dumps(entities, indent=2))
```

## Batch Processing

### Sequential Batch Processing

```python
def process_batch_sequential(prompts: List[str], model_name: str = 'gemini-1.5-pro') -> List[str]:
    """Process multiple prompts sequentially"""
    model = genai.GenerativeModel(model_name)
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            print(f"Processing {i+1}/{len(prompts)}")
            response = model.generate_content(prompt)
            results.append(response.text)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            results.append(f"Error: {str(e)}")
    
    return results

# Example batch processing
prompts = [
    "Summarize the benefits of renewable energy",
    "Explain machine learning in simple terms",
    "Write a haiku about coding",
    "List 5 Python best practices"
]

batch_results = process_batch_sequential(prompts)
for i, result in enumerate(batch_results):
    print(f"\n--- Result {i+1} ---")
    print(result[:200] + "..." if len(result) > 200 else result)
```

### Parallel Batch Processing with AsyncIO

```python
import asyncio
import aiohttp
from typing import List, Dict

class AsyncGeminiProcessor:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    async def process_single(self, prompt: str, model_name: str = 'gemini-1.5-pro') -> str:
        """Process a single prompt asynchronously"""
        async with self.semaphore:
            try:
                url = f"{self.base_url}/{model_name}:generateContent"
                headers = {'Content-Type': 'application/json'}
                
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, 
                        json=payload, 
                        headers=headers,
                        params={'key': self.api_key}
                    ) as response:
                        data = await response.json()
                        
                        if response.status == 200:
                            return data['candidates'][0]['content']['parts'][0]['text']
                        else:
                            return f"Error: {data.get('error', {}).get('message', 'Unknown error')}"
                            
            except Exception as e:
                return f"Error: {str(e)}"
    
    async def process_batch(self, prompts: List[str], model_name: str = 'gemini-1.5-pro') -> List[str]:
        """Process multiple prompts in parallel"""
        tasks = [self.process_single(prompt, model_name) for prompt in prompts]
        return await asyncio.gather(*tasks)

# Example usage
async def run_parallel_batch():
    processor = AsyncGeminiProcessor(os.environ['GOOGLE_API_KEY'])
    
    prompts = [
        "Explain photosynthesis in 100 words",
        "Write a Python function to reverse a string",
        "What are the main causes of climate change?",
        "Describe the process of DNA replication"
    ]
    
    results = await processor.process_batch(prompts)
    
    for i, result in enumerate(results):
        print(f"\n--- Parallel Result {i+1} ---")
        print(result[:150] + "..." if len(result) > 150 else result)

# Run the async batch
# asyncio.run(run_parallel_batch())
```

### Batch Processing with Progress Tracking

```python
from tqdm import tqdm
import csv

def process_csv_batch(csv_file: str, prompt_column: str, output_file: str):
    """Process a CSV file with prompts and save results"""
    
    # Read CSV
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    results = []
    
    # Process with progress bar
    for row in tqdm(rows, desc="Processing prompts"):
        try:
            prompt = row[prompt_column]
            response = model.generate_content(prompt)
            
            result_row = row.copy()
            result_row['gemini_response'] = response.text
            result_row['status'] = 'success'
            
        except Exception as e:
            result_row = row.copy()
            result_row['gemini_response'] = str(e)
            result_row['status'] = 'error'
        
        results.append(result_row)
        time.sleep(0.5)  # Rate limiting
    
    # Save results
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Processed {len(results)} items. Results saved to {output_file}")

# Example CSV processing
# process_csv_batch('input_prompts.csv', 'prompt', 'results.csv')
```

## Function Calling

### Define Functions

```python
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    # Mock weather function
    return f"The weather in {location} is sunny with 22°C"

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> Dict:
    """Calculate tip amount"""
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2)
    }

# Function schemas for Gemini
weather_schema = genai.protos.FunctionDeclaration(
    name="get_weather",
    description="Get current weather information for a location",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "location": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The city or location to get weather for"
            )
        },
        required=["location"]
    )
)

tip_schema = genai.protos.FunctionDeclaration(
    name="calculate_tip",
    description="Calculate tip amount for a bill",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "bill_amount": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="The total bill amount"
            ),
            "tip_percentage": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="The tip percentage (default 15%)"
            )
        },
        required=["bill_amount"]
    )
)
```

### Function Calling Implementation

```python
class GeminiFunctionCaller:
    def __init__(self):
        self.functions = {
            "get_weather": get_weather,
            "calculate_tip": calculate_tip
        }
        
        self.function_schemas = [weather_schema, tip_schema]
        
        self.model = genai.GenerativeModel(
            'gemini-1.5-pro',
            tools=[genai.protos.Tool(function_declarations=self.function_schemas)]
        )
    
    def call_with_functions(self, prompt: str) -> str:
        """Handle function calling conversation"""
        chat = self.model.start_chat()
        response = chat.send_message(prompt)
        
        # Check if function calls are needed
        while response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if hasattr(part, 'function_call'):
                function_call = part.function_call
                function_name = function_call.name
                function_args = dict(function_call.args)
                
                print(f"Calling function: {function_name}")
                print(f"Arguments: {function_args}")
                
                # Execute the function
                if function_name in self.functions:
                    try:
                        result = self.functions[function_name](**function_args)
                        
                        # Send function result back
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_name,
                                        response={"result": result}
                                    )
                                )]
                            )
                        )
                    except Exception as e:
                        return f"Function execution error: {str(e)}"
                else:
                    return f"Unknown function: {function_name}"
            else:
                return response.text
        
        return response.text

# Example usage
function_caller = GeminiFunctionCaller()

# Test weather function
weather_response = function_caller.call_with_functions(
    "What's the weather like in Tokyo?"
)
print("Weather Response:", weather_response)

# Test tip calculation
tip_response = function_caller.call_with_functions(
    "I have a bill of $85.50. Can you calculate a 20% tip?"
)
print("Tip Response:", tip_response)
```

## Safety Settings

### Comprehensive Safety Configuration

```python
def create_safe_model(model_name: str = 'gemini-1.5-pro'):
    """Create a model with comprehensive safety settings"""
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    return genai.GenerativeModel(
        model_name,
        safety_settings=safety_settings
    )

def check_safety_ratings(response):
    """Check safety ratings of a response"""
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'safety_ratings'):
            print("Safety Ratings:")
            for rating in candidate.safety_ratings:
                print(f"  {rating.category}: {rating.probability}")
    
    return response

# Example with safety checking
safe_model = create_safe_model()
response = safe_model.generate_content("Tell me about cybersecurity best practices")
check_safety_ratings(response)
print(response.text)
```

## Token Counting

### Count Tokens

```python
def count_tokens(text: str, model_name: str = 'gemini-1.5-pro') -> int:
    """Count tokens in text"""
    model = genai.GenerativeModel(model_name)
    return model.count_tokens(text).total_tokens

def estimate_cost(prompt: str, expected_response_length: int = 1000, model_name: str = 'gemini-1.5-pro'):
    """Estimate API cost for a request"""
    input_tokens = count_tokens(prompt, model_name)
    total_tokens = input_tokens + expected_response_length
    
    # Pricing (example rates - check current pricing)
    if 'flash' in model_name:
        cost_per_1k = 0.00075  # $0.00075 per 1K tokens
    else:
        cost_per_1k = 0.0025   # $0.0025 per 1K tokens
    
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    
    return {
        "input_tokens": input_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6)
    }

# Example usage
prompt = "Write a detailed analysis of renewable energy trends in 2024"
cost_estimate = estimate_cost(prompt, 2000)
print(f"Cost estimate: {cost_estimate}")
```

## Streaming Responses

### Real-time Streaming

```python
def stream_response(prompt: str, model_name: str = 'gemini-1.5-pro'):
    """Stream response in real-time"""
    model = genai.GenerativeModel(model_name)
    
    print("Streaming response:")
    print("-" * 50)
    
    response = model.generate_content(prompt, stream=True)
    
    full_response = ""
    for chunk in response:
        if chunk.text:
            print(chunk.text, end='', flush=True)
            full_response += chunk.text
    
    print("\n" + "-" * 50)
    return full_response

# Example streaming
streamed_text = stream_response(
    "Write a story about a time-traveling scientist who discovers that changing the past creates parallel universes"
)
```

### Streaming Chat

```python
class StreamingChat:
    def __init__(self, model_name: str = 'gemini-1.5-pro'):
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()
    
    def send_message_stream(self, message: str):
        """Send message with streaming response"""
        print(f"\nYou: {message}")
        print("Assistant: ", end='', flush=True)
        
        response = self.chat.send_message(message, stream=True)
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                print(chunk.text, end='', flush=True)
                full_response += chunk.text
        
        print()  # New line after streaming
        return full_response

# Example streaming chat
streaming_chat = StreamingChat()
streaming_chat.send_message_stream("Explain quantum computing")
streaming_chat.send_message_stream("How does it differ from classical computing?")
```

## Error Handling

### Comprehensive Error Handling

```python
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiErrorHandler:
    def __init__(self, model_name: str = 'gemini-1.5-pro'):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def safe_generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate content with error handling and retries"""
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                # Check if response was blocked
                if not response.candidates:
                    logger.warning("Response was blocked by safety filters")
                    return None
                
                candidate = response.candidates[0]
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == 'SAFETY':
                        logger.warning("Response blocked due to safety concerns")
                        return None
                    elif candidate.finish_reason == 'RECITATION':
                        logger.warning("Response blocked due to recitation")
                        return None
                
                return response.text
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    return None
        
        return None
    
    def handle_quota_exceeded(self, prompt: str):
        """Handle quota exceeded errors"""
        try:
            # Try with a more efficient model
            flash_model = genai.GenerativeModel('gemini-1.5-flash')
            response = flash_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Fallback model also failed: {str(e)}")
            return None

# Example usage
error_handler = GeminiErrorHandler()

# Test error handling
result = error_handler.safe_generate("Write a poem about artificial intelligence")
if result:
    print("Success:", result[:100] + "...")
else:
    print("Failed to generate content")
```

## Best Practices

### Performance Optimization

```python
class OptimizedGeminiClient:
    def __init__(self):
        self.models = {
            'pro': genai.GenerativeModel('gemini-1.5-pro'),
            'flash': genai.GenerativeModel('gemini-1.5-flash')
        }
        self.cache = {}
    
    def choose_model(self, task_type: str) -> str:
        """Choose appropriate model based on task"""
        if task_type in ['simple_qa', 'summarization', 'translation']:
            return 'flash'  # Faster, cheaper
        elif task_type in ['complex_reasoning', 'code_generation', 'analysis']:
            return 'pro'    # More capable
        else:
            return 'flash'  # Default to faster model
    
    def generate_with_cache(self, prompt: str, task_type: str = 'general') -> str:
        """Generate with caching for repeated prompts"""
        prompt_hash = hash(prompt)
        
        if prompt_hash in self.cache:
            print("Cache hit!")
            return self.cache[prompt_hash]
        
        model_key = self.choose_model(task_type)
        response = self.models[model_key].generate_content(prompt)
        
        result = response.text
        self.cache[prompt_hash] = result
        
        return result

# Usage tips and examples
optimized_client = OptimizedGeminiClient()

# Use appropriate model for task
summary = optimized_client.generate_with_cache(
    "Summarize the key points of machine learning",
    task_type='summarization'
)

analysis = optimized_client.generate_with_cache(
    "Provide a detailed analysis of market trends in renewable energy",
    task_type='analysis'
)
```

### Rate Limiting and Quotas

```python
import time
from collections import deque
from datetime import datetime, timedelta

class RateLimitedGemini:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
        
        # Remove old requests outside the window
        while (self.request_times and 
               self.request_times[0] < now - timedelta(minutes=1)):
            self.request_times.popleft()
        
        # Check if we need to wait
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    def generate_content(self, prompt: str) -> str:
        """Generate content with rate limiting"""
        self._wait_if_needed()
        
        # Record this request
        self.request_times.append(datetime.now())
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                print("Quota exceeded. Consider upgrading your plan.")
            raise e

# Example usage
rate_limited_client = RateLimitedGemini(requests_per_minute=30)

# This will automatically handle rate limiting
for i in range(5):
    result = rate_limited_client.generate_content(f"Write a short fact about topic {i}")
    print(f"Fact {i}: {result[:100]}...")
```

## Advanced Features

### Content Filtering and Moderation

```python
def moderate_content(text: str) -> Dict:
    """Check if content needs moderation"""
    
    moderation_prompt = f"""
    Analyze the following text for potential issues:
    
    Text: {text}
    
    Check for:
    1. Inappropriate content
    2. Potential misinformation
    3. Harmful instructions
    4. Privacy violations
    
    Respond with a JSON object containing:
    - is_safe: boolean
    - concerns: array of strings
    - severity: "low", "medium", or "high"
    - recommendation: string
    """
    
    return generate_structured_output(moderation_prompt, {
        "type": "object",
        "properties": {
            "is_safe": {"type": "boolean"},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "recommendation": {"type": "string"}
        }
    })

# Example moderation
user_input = "How do I bake a chocolate cake?"
moderation_result = moderate_content(user_input)
print(json.dumps(moderation_result, indent=2))
```

### Context Management for Long Conversations

```python
class ContextManagedChat:
    def __init__(self, max_context_length: int = 30000):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.context_summary = ""
    
    def _get_context_length(self, text: str) -> int:
        """Estimate context length"""
        return len(text.split())  # Rough token estimation
    
    def _summarize_old_context(self):
        """Summarize old conversation when context gets too long"""
        if len(self.conversation_history) > 10:
            old_messages = self.conversation_history[:-5]  # Keep last 5 messages
            
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in old_messages
            ])
            
            summary_prompt = f"""
            Summarize the key points and context from this conversation:
            
            {conversation_text}
            
            Focus on important decisions, facts established, and ongoing topics.
            """
            
            response = self.model.generate_content(summary_prompt)
            self.context_summary = response.text
            
            # Keep only recent messages
            self.conversation_history = self.conversation_history[-5:]
    
    def send_message(self, message: str) -> str:
        """Send message with context management"""
        
        # Check if we need to summarize old context
        total_context = self.context_summary + "\n".join([
            msg['content'] for msg in self.conversation_history
        ]) + message
        
        if self._get_context_length(total_context) > self.max_context_length:
            self._summarize_old_context()
        
        # Build context for the request
        context = ""
        if self.context_summary:
            context += f"Previous conversation summary: {self.context_summary}\n\n"
        
        context += "Recent conversation:\n"
        for msg in self.conversation_history[-5:]:
            context += f"{msg['role']}: {msg['content']}\n"
        
        context += f"User: {message}\nAssistant: "
        
        # Generate response
        response = self.model.generate_content(context)
        
        # Update history
        self.conversation_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.text}
        ])
        
        return response.text

# Example long conversation management
managed_chat = ContextManagedChat()

# Simulate a long conversation
topics = [
    "Tell me about machine learning",
    "How does neural network training work?", 
    "What are the different types of ML algorithms?",
    "Explain deep learning architectures",
    "How do I choose the right algorithm for my problem?"
]

for topic in topics:
    response = managed_chat.send_message(topic)
    print(f"Q: {topic}")
    print(f"A: {response[:150]}...\n")
```

### Multi-modal Processing Pipeline

```python
class MultiModalProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def process_document(self, image_path: str, text_content: str, analysis_type: str) -> Dict:
        """Process a document with both image and text"""
        
        image = PIL.Image.open(image_path)
        
        prompts = {
            "summary": "Provide a comprehensive summary of this document",
            "key_points": "Extract the key points and main arguments",
            "analysis": "Analyze the data and provide insights",
            "questions": "Generate questions that this document answers"
        }
        
        prompt = f"""
        Document Content: {text_content}
        
        Task: {prompts.get(analysis_type, prompts['summary'])}
        
        Please analyze both the text content and the image to provide a complete response.
        """
        
        response = self.model.generate_content([prompt, image])
        
        return {
            "analysis_type": analysis_type,
            "result": response.text,
            "timestamp": datetime.now().isoformat()
        }
    
    def compare_documents(self, doc_paths: List[str], comparison_focus: str) -> str:
        """Compare multiple documents"""
        
        images = [PIL.Image.open(path) for path in doc_paths]
        
        prompt = f"""
        Compare these {len(doc_paths)} documents focusing on: {comparison_focus}
        
        Provide:
        1. Similarities between the documents
        2. Key differences
        3. Unique insights from each
        4. Overall synthesis
        """
        
        content = [prompt] + images
        response = self.model.generate_content(content)
        
        return response.text

# Example multi-modal processing
processor = MultiModalProcessor()

# Process a single document
# result = processor.process_document(
#     "report.png", 
#     "This is the text content of the report...",
#     "analysis"
# )

# Compare multiple documents
# comparison = processor.compare_documents(
#     ["doc1.png", "doc2.png", "doc3.png"],
#     "financial performance trends"
# )
```

## Conclusion

This comprehensive guide covers the major capabilities of the Gemini API:

- **Basic Usage**: Text generation and chat functionality
- **Advanced Features**: Vision, function calling, structured output
- **Production Considerations**: Error handling, rate limiting, cost optimization
- **Specialized Use Cases**: Batch processing, content moderation, context management

### Next Steps

1. **Set up your API key** and test basic functionality
2. **Choose the right model** for your use case (Pro vs Flash)
3. **Implement error handling** and rate limiting for production
4. **Optimize costs** by using appropriate models and caching
5. **Monitor usage** and adjust quotas as needed

### Additional Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python SDK Reference](https://ai.google.dev/api/python)
- [Community Examples](https://github.com/google/generative-ai-python)

Remember to always follow ethical AI practices, implement proper safety measures, and respect rate limits when using the Gemini API in production applications.