app = Flask(__name__)
import os
import json
import logging
import requests
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from models import db, PromptLog
from news_scraper import (
    get_all_headlines, format_headlines_for_context, 
    search_web, format_search_results_for_context,
    is_sports_score_query, get_mlb_scores, format_mlb_scores_for_display
)
from multimedia_generator import generate_text_to_speech, generate_text_to_image, generate_text_to_video
from openrouter_api import OpenRouterAPI

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Configure the SQLAlchemy database connection
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the database
db.init_app(app)

# Create all tables
with app.app_context():
    db.create_all()

# Define the available models
MODELS = {
    "anthropic/claude-3-opus-20240229": "Claude 3 Opus",
    "anthropic/claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
    "google/gemini-pro": "Gemini Pro",
    "meta-llama/llama-3-70b-instruct": "Llama 3 70B",
    "deepseek/deepseek-coder": "DeepSeek Coder"
}

def get_api_key():
    """Get OpenRouter API key from environment or config file"""
    # First try to get from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    logger.info(f"API key from environment: {'Found' if api_key else 'Not found'}")

    # Try loading from config.json if environment variable is not set
    if not api_key:
        try:
            with open('config.json', 'r') as config_file:
                config = json.load(config_file)
                api_key = config.get('OPENROUTER_API_KEY')
                logger.info(f"API key from config.json: {'Found' if api_key else 'Not found'}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config.json: {str(e)}")

    return api_key



# Save log functionality has been integrated directly into route handlers

def is_news_related_query(prompt):
    """Check if the prompt is asking for news or current events"""
    news_patterns = [
        r'news',
        r'headlines',
        r'current events',
        r'latest',
        r'today\'s',
        r'recent',
        r'what\'s happening',
        r'update me',
    ]

    # Case insensitive check for news-related terms
    for pattern in news_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True
    return False

def is_search_query(prompt):
    """Check if the prompt is asking for information that would benefit from web search"""
    # Detect search intent patterns
    search_patterns = [
        r'search for',
        r'look up',
        r'find information',
        r'find out about',
        r'what is',
        r'who is',
        r'where is',
        r'when is',
        r'how to',
        r'why is',
        r'tell me about',
        r'information about',
        r'facts about',
        r'research',
    ]

    # Case insensitive check for search-related terms
    for pattern in search_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True

    # Check for question patterns that suggest search
    if re.search(r'^(what|who|where|when|why|how)', prompt, re.IGNORECASE):
        return True

    return False

def extract_search_query(prompt):
    """Extract the actual search query from the user prompt"""
    # Extract query from common patterns like "search for X" or "find information about X"
    patterns = [
        r'search for (.*)',
        r'look up (.*)',
        r'find information (about|on) (.*)',
        r'find out about (.*)',
        r'what is (.*)',
        r'who is (.*)',
        r'where is (.*)',
        r'when is (.*)',
        r'how to (.*)',
        r'why is (.*)',
        r'tell me about (.*)',
        r'information about (.*)',
        r'facts about (.*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            if len(match.groups()) > 1:
                return match.group(2)  # For patterns with multiple capture groups
            return match.group(1)

    # If no pattern matches, use the whole prompt as the search query
    return prompt

def enrich_prompt_with_search(prompt):
    """Perform a web search and add results to the prompt"""
    logger.info("Performing web search to enhance prompt...")

    # Extract the core search query from the prompt
    search_query = extract_search_query(prompt)
    logger.info(f"Extracted search query: {search_query}")

    # Perform the search
    search_results = search_web(search_query)
    search_context = format_search_results_for_context(search_query, search_results)

    enhanced_prompt = f"""
{search_context}

Based on the above search results, please provide a helpful response to the following user query:

{prompt}
"""
    logger.info("Successfully enriched the prompt with web search results")
    return enhanced_prompt

def enrich_prompt_with_news(prompt):
    """Fetch news headlines and add them to the prompt"""
    logger.info("Fetching current news headlines for context...")
    headlines = get_all_headlines()
    news_context = format_headlines_for_context(headlines)

    enhanced_prompt = f"""
{news_context}

Based on the above current news headlines, please respond to the following user query:

{prompt}
"""
    logger.info("Successfully enriched the prompt with current news headlines")
    return enhanced_prompt

def send_prompt_to_model(prompt, model_id):
    """Send prompt to the selected model using OpenRouter API"""
    api_key = get_api_key()

    if not api_key:
        return False, "OpenRouter API key is missing. Please set the OPENROUTER_API_KEY environment variable or add it to config.json."

    # Check if the prompt is asking for sports scores (especially MLB)
    if is_sports_score_query(prompt):
        logger.info("Detected sports scores query. Fetching MLB scores.")
        mlb_data = get_mlb_scores()
        if mlb_data:
            formatted_scores = format_mlb_scores_for_display(mlb_data)
            logger.info("Successfully fetched MLB scores")
            return True, formatted_scores, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Check if the prompt is asking for news or requires search
    original_prompt = prompt
    if is_news_related_query(prompt):
        logger.info("Detected news-related query. Enhancing with current headlines.")
        prompt = enrich_prompt_with_news(prompt)
    elif is_search_query(prompt):
        logger.info("Detected search-related query. Enhancing with web search results.")
        prompt = enrich_prompt_with_search(prompt)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://passivestack.store",  # Site URL for better request tracking
        "X-Title": "Universal AI Prompt Portal"  # App name for better request tracking
    }

    # Create a more detailed system prompt that directs users to our internal tools
    system_prompt = """You are a helpful AI assistant responding to a user through the Universal AI Prompt Portal at passivestack.store.
This portal has enhanced your capabilities with access to current news headlines and web search results when needed.

IMPORTANT: DO NOT refer users to external tools or websites for image, audio, or video generation. Instead:
1. For image generation requests, tell users to click the 'Image Generation' link in the portal navigation or visit "/image_generator" 
2. For text-to-speech requests, tell users to click the 'Text to Speech' link in the portal navigation or visit "/text_to_speech_page"
3. For video generation requests, tell users to click the 'Video Generation' link in the portal navigation or visit "/video_generator_page"

Answer the user's questions and follow their instructions to the best of your ability. Provide helpful, accurate, and thoughtful responses based on the information provided to you.
"""

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150,  # Reduced further to avoid credit limit issues
        "stream": False
    }

    try:
        logger.info(f"Sending prompt to {model_id}")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )

        # Check for credit limit error first, before trying to parse the response
        if response.status_code == 402:
            response_text = response.text
            logger.info(f"Credit limit error response: {response_text}")

            # Extract error message if available
            error_message = "Not enough credits in OpenRouter account. Please reduce token count or upgrade your OpenRouter account."
            try:
                response_data = response.json()
                if 'error' in response_data and 'message' in response_data['error']:
                    error_message = response_data['error']['message']
            except Exception as e:
                logger.error(f"Failed to parse credit limit error: {str(e)}")

            return False, error_message, None

        elif response.status_code == 200:
            response_text = response.text
            logger.info(f"Raw API Response: {response_text}")

            try:
                response_data = response.json()
                logger.info(f"Response JSON: {json.dumps(response_data, indent=2)}")

                # Check for error in the response (sometimes API returns 200 but still has an error)
                if 'error' in response_data:
                    error_message = "API returned an error"
                    if isinstance(response_data['error'], dict) and 'message' in response_data['error']:
                        error_message = response_data['error']['message']
                    elif isinstance(response_data['error'], str):
                        error_message = response_data['error']

                    logger.error(f"API returned error with status 200: {error_message}")
                    return False, error_message, None

                # The API response might be in different formats
                # Let's handle different possible formats

                # Try OpenAI format first
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    choice = response_data['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        ai_response = choice['message']['content']
                    elif 'text' in choice:
                        # Some APIs return text directly
                        ai_response = choice['text']
                    else:
                        logger.error(f"Unexpected response format: {choice}")
                        return False, "Error: Unknown response format from API", None
                # Try Anthropic format
                elif 'completion' in response_data:
                    ai_response = response_data['completion']
                # Try simple format
                elif 'output' in response_data:
                    ai_response = response_data['output']
                # Response has a different format
                else:
                    logger.error(f"Unrecognized API response format: {str(response_data)}")
                    return False, "The API response format is not recognized. Please check if the OpenRouter API has changed.", None

                # Log successful response details
                usage = response_data.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)

                logger.info(f"Successfully received response from {model_id}. Tokens used: {total_tokens}")

                return True, ai_response, {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            except Exception as parse_error:
                logger.error(f"Error parsing API response: {str(parse_error)}")
                logger.error(f"Response text: {response_text}")
                return False, f"Error processing API response: {str(parse_error)}", None
        elif response.status_code == 401 or response.status_code == 403:
            logger.error(f"Authentication error: {response.status_code}")
            return False, "API key invalid or expired. OpenRouter API keys should start with 'sk-or-'. Please check your API key.", None
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            return False, "Rate limit exceeded. Please try again later.", None
        elif response.status_code >= 500:
            logger.error(f"OpenRouter server error: {response.status_code}")
            return False, "OpenRouter service is experiencing issues. Please try again later.", None
        else:
            error_detail = "Unknown error"
            try:
                error_json = response.json()
                if 'error' in error_json:
                    if isinstance(error_json['error'], dict) and 'message' in error_json['error']:
                        error_detail = error_json['error']['message']
                    elif isinstance(error_json['error'], str):
                        error_detail = error_json['error']
            except Exception:
                pass

            logger.error(f"API request failed with status code {response.status_code}: {error_detail}")
            return False, f"Error: {error_detail}", None

    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return False, "Request to OpenRouter API timed out. The server might be busy, please try again.", None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return False, "Could not connect to OpenRouter API. Please check your internet connection.", None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return False, f"Error connecting to OpenRouter API: {str(e)}", None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"Unexpected error: {str(e)}", None

@app.route('/')
def index():
    """Main landing page"""
    api_key = get_api_key()
    has_api_key = bool(api_key)

    # Check if the user is on a mobile device
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_agents = ['android', 'webos', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone']

    # Determine if it's a mobile device
    is_mobile = any(agent in user_agent for agent in mobile_agents)

    # Check for explicit mobile preference in query param or cookie
    mobile_param = request.args.get('mobile', '')
    mobile_cookie = request.cookies.get('mobile_preference', '')

    # If mobile param is provided, set cookie and use that preference
    if mobile_param in ['1', 'true', 'yes']:
        is_mobile = True
    elif mobile_param in ['0', 'false', 'no']:
        is_mobile = False
    # Otherwise use cookie if available
    elif mobile_cookie in ['1', 'true', 'yes']:
        is_mobile = True
    elif mobile_cookie in ['0', 'false', 'no']:
        is_mobile = False

    # Force mobile view for testing if needed
    # is_mobile = True  # Uncomment to force mobile view

    # Define model information for mobile UI
    mobile_models = {
        'anthropic/claude-3-sonnet-20240229': {
            'display_name': 'Claude 3',
            'gradient': 'linear-gradient(to bottom right, #9333ea, #6366f1)',
            'description': 'Anthropic\'s latest model, powerful for complex reasoning and creative tasks.',
            'icon': 'fa-solid fa-brain'
        },
        'openai/gpt-3.5-turbo': {
            'display_name': 'GPT-3.5',
            'gradient': 'linear-gradient(to bottom right, #f59e0b, #ef4444)',
            'description': 'OpenAI\'s efficient model, fast responses for general queries.',
            'icon': 'fa-solid fa-bolt'
        },
        'google/gemini-pro': {
            'display_name': 'Gemini Pro',
            'gradient': 'linear-gradient(to bottom right, #34d399, #0ea5e9)',
            'description': 'Google\'s advanced model with multimodal understanding.',
            'icon': 'fa-solid fa-gem'
        },
        'meta-llama/llama-3-70b-instruct': {
            'display_name': 'Llama 3',
            'gradient': 'linear-gradient(to bottom right, #8b5cf6, #ec4899)',
            'description': 'Meta\'s open model with impressive capabilities.',
            'icon': 'fa-solid fa-dragon'
        }
    }

    # Get AI-powered prompt suggestions
    try:
        from prompt_suggestions import get_mixed_suggestions, get_cached_suggestions

        # Try to get cached suggestions first
        cached = get_cached_suggestions()
        if cached and 'mixed' in cached:
            prompt_suggestions = cached['mixed'][:5]  # Limit to 5 suggestions
        else:
            # Generate new suggestions if cache unavailable
            prompt_suggestions = get_mixed_suggestions(count=5)

        logger.info(f"Generated {len(prompt_suggestions)} prompt suggestions")
    except Exception as e:
        logger.error(f"Failed to load prompt suggestions: {str(e)}")
        prompt_suggestions = [
            "Explain the concept of neural networks",
            "Write a short story about a space explorer",
            "Compare different renewable energy sources",
            "What are the best practices for remote work?",
            "How can artificial intelligence help with climate change?"
        ]

    if is_mobile:
        # Use the mobile template
        from flask import make_response
        response = make_response(render_template(
            'mobile_index.html', 
            models=mobile_models,
            selected_model='anthropic/claude-3-sonnet-20240229',
            conversation_history=[],
            has_api_key=has_api_key,
            prompt_suggestions=prompt_suggestions
        ))
        # Set cookie for future visits if coming from mobile param
        if mobile_param:
            response.set_cookie('mobile_preference', '1' if is_mobile else '0', max_age=30*24*60*60)
        return response
    else:
        # Use the desktop template
        return render_template(
            'index.html',
            models=MODELS,
            has_api_key=has_api_key,
            prompt_suggestions=prompt_suggestions
        )

@app.route('/image_generator')
def image_generator():
    """Image generation page"""
    api_key = get_api_key()
    has_api_key = bool(api_key)

    return render_template(
        'image_generator.html',
        has_api_key=has_api_key
    )

@app.route('/search_page')
def search_page():
    """Web search page"""
    query = request.args.get('q', '')
    results = []

    if query:
        try:
            results = search_web(query)
        except Exception as e:
            flash(f"Error performing search: {str(e)}", "error")

    return render_template(
        'search.html',
        query=query,
        results=results
    )

@app.route('/text_to_speech_page')
def text_to_speech_page():
    """Text to speech page"""
    return render_template('text_to_speech.html')

@app.route('/video_generator_page')
def video_generator_page():
    """Video generator page"""
    return render_template('video_generator.html')

@app.route('/mobile')
def mobile_index():
    """Dedicated mobile landing page"""
    api_key = get_api_key()
    has_api_key = bool(api_key)

    # Define model information for mobile UI
    mobile_models = {
        'anthropic/claude-3-sonnet-20240229': {
            'display_name': 'Claude 3',
            'gradient': 'linear-gradient(to bottom right, #9333ea, #6366f1)',
            'description': 'Anthropic\'s latest model, powerful for complex reasoning and creative tasks.',
            'icon': 'fa-solid fa-brain'
        },
        'openai/gpt-3.5-turbo': {
            'display_name': 'GPT-3.5',
            'gradient': 'linear-gradient(to bottom right, #f59e0b, #ef4444)',
            'description': 'OpenAI\'s efficient model, fast responses for general queries.',
            'icon': 'fa-solid fa-bolt'
        },
        'google/gemini-pro': {
            'display_name': 'Gemini Pro',
            'gradient': 'linear-gradient(to bottom right, #34d399, #0ea5e9)',
            'description': 'Google\'s advanced model with multimodal understanding.',
            'icon': 'fa-solid fa-gem'
        },
        'meta-llama/llama-3-70b-instruct': {
            'display_name': 'Llama 3',
            'gradient': 'linear-gradient(to bottom right, #8b5cf6, #ec4899)',
            'description': 'Meta\'s open model with impressive capabilities.',
            'icon': 'fa-solid fa-dragon'
        }
    }

    # Get AI-powered prompt suggestions
    try:
        from prompt_suggestions import get_mixed_suggestions, get_cached_suggestions

        # Try to get cached suggestions first
        cached = get_cached_suggestions()
        if cached and 'mixed' in cached:
            prompt_suggestions = cached['mixed'][:5]  # Limit to 5 suggestions
        else:
            # Generate new suggestions if cache unavailable
            prompt_suggestions = get_mixed_suggestions(count=5)

        logger.info(f"Generated {len(prompt_suggestions)} prompt suggestions")
    except Exception as e:
        logger.error(f"Failed to load prompt suggestions: {str(e)}")
        prompt_suggestions = [
            "Explain the concept of neural networks",
            "Write a short story about a space explorer",
            "Compare different renewable energy sources",
            "What are the best practices for remote work?",
            "How can artificial intelligence help with climate change?"
        ]

    # Use the mobile template
    from flask import make_response
    response = make_response(render_template(
        'mobile_index.html', 
        models=mobile_models,
        selected_model='anthropic/claude-3-sonnet-20240229',
        conversation_history=[],
        has_api_key=has_api_key,
        prompt_suggestions=prompt_suggestions
    ))
    # Set cookie for mobile preference
    response.set_cookie('mobile_preference', '1', max_age=30*24*60*60)
    return response

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    """Handle prompt submission"""
    prompt = request.form.get('prompt', '')
    model_id = request.form.get('model', 'anthropic/claude-3-sonnet-20240229')

    # Determine if this is mobile or API request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    is_api = request.headers.get('Accept', '').startswith('application/json')

    # Check if the user is on a mobile device
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_agents = ['android', 'webos', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone']

    # Determine if it's a mobile device
    is_mobile = any(agent in user_agent for agent in mobile_agents)
    mobile_cookie = request.cookies.get('mobile_preference', '')

    # Override mobile detection if there's a cookie preference
    if mobile_cookie == '1':
        is_mobile = True
    elif mobile_cookie == '0':
        is_mobile = False

    # Force JSON response for API requests
    if is_ajax or is_api:
        is_mobile = False

    # Validate inputs
    if not prompt:
        if is_mobile:
            flash("Prompt cannot be empty", "error")
            return redirect(url_for('index'))
        return jsonify({"success": False, "error": "Prompt cannot be empty"})

    if model_id not in MODELS:
        if is_mobile:
            flash(f"Invalid model selection: {model_id}", "error")
            return redirect(url_for('index'))
        return jsonify({"success": False, "error": f"Invalid model selection: {model_id}"})

    # Send to model
    success, response, token_info = send_prompt_to_model(prompt, model_id)

    if success:
        # Get token information from API response or use estimation
        if token_info:
            prompt_tokens = token_info['prompt_tokens']
            response_tokens = token_info['completion_tokens']
            total_tokens = token_info['total_tokens']
        else:
            # Fallback to estimation (1 token ≈ 4 characters)
            prompt_tokens = len(prompt) // 4
            response_tokens = len(response) // 4
            total_tokens = prompt_tokens + response_tokens

        # Create log entry
        log_entry = PromptLog(
            model=model_id,
            prompt=prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens
        )

        # Add to database session and commit
        db.session.add(log_entry)
        db.session.commit()

        # Check if we want a mobile view or API response
        if is_mobile:
            # Define model information for mobile UI
            mobile_models = {
                'anthropic/claude-3-sonnet-20240229': {
                    'display_name': 'Claude 3',
                    'gradient': 'linear-gradient(to bottom right, #9333ea, #6366f1)',
                    'description': 'Anthropic\'s latest model, powerful for complex reasoning and creative tasks.',
                    'icon': 'fa-solid fa-brain'
                },
                'openai/gpt-3.5-turbo': {
                    'display_name': 'GPT-3.5',
                    'gradient': 'linear-gradient(to bottom right, #f59e0b, #ef4444)',
                    'description': 'OpenAI\'s efficient model, fast responses for general queries.',
                    'icon': 'fa-solid fa-bolt'
                },
                'google/gemini-pro': {
                    'display_name': 'Gemini Pro',
                    'gradient': 'linear-gradient(to bottom right, #34d399, #0ea5e9)',
                    'description': 'Google\'s advanced model with multimodal understanding.',
                    'icon': 'fa-solid fa-gem'
                },
                'meta-llama/llama-3-70b-instruct': {
                    'display_name': 'Llama 3',
                    'gradient': 'linear-gradient(to bottom right, #8b5cf6, #ec4899)',
                    'description': 'Meta\'s open model with impressive capabilities.',
                    'icon': 'fa-solid fa-dragon'
                }
            }

            # Get follow-up prompt suggestions
            try:
                from prompt_suggestions import get_personalized_suggestions

                # Generate follow-up suggestions based on the conversation
                followup_suggestions = get_personalized_suggestions(count=5)

                # Add some context-specific suggestions based on the model's response
                # This could be enhanced with more advanced NLP in the future

                logger.info(f"Generated {len(followup_suggestions)} follow-up suggestions")
            except Exception as e:
                logger.error(f"Failed to load follow-up suggestions: {str(e)}")
                followup_suggestions = [
                    "Can you explain that in more detail?",
                    "What are some practical applications of this?",
                    "How does this compare to alternative approaches?",
                    "What are the limitations to consider?",
                    "Can you provide some examples?"
                ]

            # Return the mobile template with the conversation and suggestions
            return render_template(
                'mobile_prompt_result.html',
                models=mobile_models,
                model=model_id,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                prompt_suggestions=followup_suggestions
            )
        else:
            # Return a JSON response for API clients
            return jsonify({
                "success": True, 
                "response": response,
                "tokens": {
                    "prompt": prompt_tokens,
                    "response": response_tokens,
                    "total": total_tokens
                }
            })
    else:
        # Error handling
        if is_mobile:
            flash(f"Error: {response}", "error")
            return redirect(url_for('index'))
        else:
            # Return error message as JSON
            return jsonify({"success": False, "error": response})

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify API connectivity"""
    model_id = request.args.get('model', 'anthropic/claude-3-sonnet-20240229')

    # Validate model selection
    if model_id not in MODELS:
        return render_template(
            'index.html',
            models=MODELS,
            error=f"Invalid model selection: {model_id}"
        )

    test_prompt = "This is a test of the Universal AI Prompt Portal. Please provide a brief greeting and confirm that you're working properly."
    success, response, token_info = send_prompt_to_model(test_prompt, model_id)

    if success:
        # Get token information from API response or use estimation
        if token_info:
            prompt_tokens = token_info['prompt_tokens']
            response_tokens = token_info['completion_tokens']
            total_tokens = token_info['total_tokens']
        else:
            # Fallback to estimation (1 token ≈ 4 characters)
            prompt_tokens = len(test_prompt) // 4
            response_tokens = len(response) // 4
            total_tokens = prompt_tokens + response_tokens

        # Create log entry
        log_entry = PromptLog(
            model=model_id,
            prompt=test_prompt,
            response=response,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens
        )

        # Add to database session and commit
        db.session.add(log_entry)
        db.session.commit()

        return render_template(
            'index.html',
            models=MODELS,
            test_result=response,
            test_model=MODELS[model_id]
        )
    else:
        return render_template(
            'index.html',
            models=MODELS,
            error=response
        )

@app.route('/search')
def search_endpoint():
    """Endpoint to perform web searches directly"""
    query = request.args.get('q', '')

    if not query:
        return jsonify({"success": False, "error": "Search query cannot be empty"})

    try:
        # Check if we have a cached result
        cache_key = f"search_{query.lower().replace(' ', '_')}"

        # Import ReplitDB
        from replit_db import ReplitDB

        # Try to get results from cache first (TTL: 1 hour = 3600 seconds)
        cached_results = ReplitDB.cache_get(cache_key)

        if cached_results:
            logger.info(f"Using cached search results for query: {query}")
            results = cached_results
            # Track cache hit
            ReplitDB.increment_stat("search_cache", "hits")
        else:
            # Perform web search
            logger.info(f"Performing new web search for query: {query}")
            results = search_web(query)

            # Cache the results for future use if we got results
            if results:
                ReplitDB.cache_set(cache_key, results, ttl=3600)  # Cache for 1 hour

            # Track cache miss
            ReplitDB.increment_stat("search_cache", "misses")

        # Track total searches
        ReplitDB.increment_stat("search", "total")

        if results:
            return jsonify({
                "success": True,
                "query": query,
                "results": results,
                "cached": cached_results is not None
            })
        else:
            return jsonify({
                "success": False,
                "error": f"No results found for query: {query}"
            })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error performing search: {str(e)}"
        })

@app.route('/api/news')
def get_news_endpoint():
    """API endpoint to get the latest news headlines"""
    try:
        # Import ReplitDB
        from replit_db import ReplitDB

        # News changes frequently, but we can still cache for a short time (15 minutes)
        cache_key = "news_headlines"

        # Try to get from cache first
        cached_headlines = ReplitDB.cache_get(cache_key)

        if cached_headlines:
            logger.info("Using cached news headlines")
            headlines = cached_headlines
            formatted_headlines = format_headlines_for_context(headlines)
            # Track cache hit
            ReplitDB.increment_stat("news_cache", "hits")
        else:
            logger.info("Fetching fresh news headlines")
            headlines = get_all_headlines()

            # Cache the results
            ReplitDB.cache_set(cache_key, headlines, ttl=900)  # Cache for 15 minutes

            formatted_headlines = format_headlines_for_context(headlines)

            # Track cache miss
            ReplitDB.increment_stat("news_cache", "misses")

        # Track total news API calls
        ReplitDB.increment_stat("news", "total")

        # Remove the AI context instructions from the formatted headlines
        cleaned_headlines = formatted_headlines.replace("Based on the above current news headlines, please respond to the following user query:", "").strip()

        return jsonify({
            "success": True,
            "headlines": cleaned_headlines,
            "cached": cached_headlines is not None
        })
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error fetching news: {str(e)}"
        })

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    """API endpoint for converting text to speech"""
    # Get data from request
    data = request.get_json() or {}
    text = data.get('text', '')
    language = data.get('language', 'en')

    if not text:
        return jsonify({"success": False, "error": "Text cannot be empty"})

    # Generate speech
    result = generate_text_to_speech(text, language)

    return jsonify(result)

@app.route('/api/text-to-image', methods=['POST'])
def text_to_image_endpoint():
    """API endpoint for converting text to image"""
    # Get data from request
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    width = int(data.get('width', 1024))
    height = int(data.get('height', 1024))
    use_ai = data.get('use_ai', True)  # Whether to try AI image generation first

    if not prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty"})

    # Check if OpenRouter API key is available
    api_key = get_api_key()

    # Log the availability of the API key
    if api_key:
        logger.info("OpenRouter API key is available for image generation")
    else:
        logger.warning("No OpenRouter API key available for AI image generation")
        use_ai = False  # Disable AI generation if no API key

    # Generate the image
    # Our updated function now handles both AI and local generation
    logger.info(f"Generating image for prompt: {prompt[:50]}...")
    result = generate_text_to_image(prompt, width, height, use_ai=use_ai)

    # Add additional info if needed
    if result.get("success") and not result.get("source"):
        result["source"] = "local_generator"

    return jsonify(result)

@app.route('/api/text-to-video', methods=['POST'])
def text_to_video_endpoint():
    """API endpoint for converting text to video"""
    # Get data from request
    data = request.get_json() or {}
    text = data.get('text', '')
    duration = int(data.get('duration', 5))

    if not text:
        return jsonify({"success": False, "error": "Text cannot be empty"})

    # Generate video
    result = generate_text_to_video(text, duration)

    return jsonify(result)

@app.route('/media/<path:filename>')
def serve_media(filename):
    """Serve media files (audio, images, videos)"""
    # Determine the correct directory based on file extension
    if filename.startswith('audio/'):
        return send_from_directory('static', filename)
    elif filename.startswith('images/'):
        return send_from_directory('static', filename)
    elif filename.startswith('videos/'):
        return send_from_directory('static', filename)
    else:
        return jsonify({"success": False, "error": "Invalid media path"}), 404

@app.route('/dashboard')
def dashboard():
    """Admin dashboard to view logs"""
    # Get filter parameters
    model_filter = request.args.get('model', '')
    search_query = request.args.get('search', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    token_min = request.args.get('token_min', '')
    token_max = request.args.get('token_max', '')

    # Query logs from database
    query = PromptLog.query

    # Apply filters if needed
    if model_filter:
        query = query.filter(PromptLog.model == model_filter)

    if search_query:
        search_term = f"%{search_query}%"
        query = query.filter(
            db.or_(
                PromptLog.prompt.ilike(search_term),
                PromptLog.response.ilike(search_term)
            )
        )

    # Date range filtering
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(PromptLog.timestamp >= from_date)
        except ValueError:
            flash(f"Invalid 'from' date format: {date_from}. Use YYYY-MM-DD format.", "error")

    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d')
            # Add one day to include the end date fully
            to_date = to_date.replace(hour=23, minute=59, second=59)
            query = query.filter(PromptLog.timestamp <= to_date)
        except ValueError:
            flash(f"Invalid 'to' date format: {date_to}. Use YYYY-MM-DD format.", "error")

    # Token count filtering
    if token_min and token_min.isdigit():
        query = query.filter(PromptLog.total_tokens >= int(token_min))

    if token_max and token_max.isdigit():
        query = query.filter(PromptLog.total_tokens <= int(token_max))

    # Sort logs by timestamp, newest first
    query = query.order_by(PromptLog.timestamp.desc())

    # Execute query and get logs
    logs = [log.to_dict() for log in query.all()]

    return render_template(
        'dashboard.html',
        logs=logs,
        models=MODELS,
        current_filter=model_filter,
        search_query=search_query,
        date_from=date_from,
        date_to=date_to,
        token_min=token_min,
        token_max=token_max
    )

@app.route('/service-status')
def service_status():
    """Display the status of all configured services"""
    from service_connector import ServiceConnector

    # Get all service statuses
    service_statuses = ServiceConnector.check_all_services()

    # Format service status info
    services = []
    for service_name, status in service_statuses.items():
        services.append({
            'name': service_name.title(),
            'status': status['status'],
            'message': status['message']
        })

    # Get API usage statistics
    from replit_db import ReplitDB
    stats = ReplitDB.get_all_stats()

    # Log all environment variables for debugging (without exposing actual values)
    env_vars = []
    for key in os.environ:
        if "KEY" in key or "TOKEN" in key or "SECRET" in key or "PASSWORD" in key:
            env_vars.append(f"{key}: [REDACTED]")
        else:
            env_vars.append(f"{key}: {os.environ[key]}")

    logger.debug(f"Available environment variables: {len(env_vars)}")

    return render_template(
        'service_status.html',
        services=services,
        stats=stats
    )

@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variables (ADMIN USE ONLY)"""
    # Safe debugging - show only which keys exist, not their values
    env_info = {}

    # Check for specific API keys
    api_keys = [
        "OPENROUTER_API_KEY", 
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "PERPLEXITY_API_KEY",
        "MISTRAL_API_KEY"
    ]

    for key in api_keys:
        value = os.environ.get(key)
        env_info[key] = "Available" if value else "Not set"

    # Test OpenRouter specifically
    try:
        from openrouter_api import OpenRouterAPI
        api_key = get_api_key()
        client = OpenRouterAPI(api_key)

        test_connection = {
            "status": "Unknown"
        }

        if client and api_key:
            # Simple validation without making an actual API call
            test_connection = {
                "status": "API Key Found",
                "key_length": len(api_key)
            }
        else:
            test_connection = {
                "status": "API Key Not Found",
                "get_api_key_returns": "None" if api_key is None else "Empty string" if api_key == "" else f"Value of length {len(api_key)}"
            }
    except Exception as e:
        test_connection = {
            "status": "Error", 
            "error": str(e)
        }

    return jsonify({
        "environment_keys": env_info,
        "openrouter_test": test_connection
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)
