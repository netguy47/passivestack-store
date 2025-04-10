import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OpenRouterAPI:
    """
    A client for interacting with the OpenRouter API to access various AI models
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter API client
        
        Args:
            api_key: OpenRouter API key. If not provided, will try to load from environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("No OpenRouter API key provided or found in environment.")
            raise ValueError("OpenRouter API key is required")
        elif not self.api_key.startswith('sk-or-'):
            logger.error("Invalid OpenRouter API key format. Key must start with 'sk-or-'")
            raise ValueError("Invalid OpenRouter API key format. Please ensure your key starts with 'sk-or-'")
            
        logger.info(f"Initializing OpenRouter client with valid key format (length: {len(self.api_key)})")
        self.base_url = "https://openrouter.ai/api/v1"
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        
        # Use OpenRouter for image generation, proxying to OpenAI/Stability
        self.images_url = f"{self.base_url}/images/generations"
        
        # Set up default headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://passivestack.store"),
            "X-Title": "Universal AI Prompt Portal"
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from OpenRouter
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(self.models_url, headers=self.headers)
            response.raise_for_status()
            
            models_data = response.json()
            logger.info(f"Successfully retrieved {len(models_data.get('data', []))} models")
            
            # Return the list of models
            return models_data.get("data", [])
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def chat_completion(self, 
                         messages: List[Dict[str, str]], 
                         model_id: str,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: Optional[int] = 300,  # Default to 300 to avoid credit limits
                         stream: bool = False) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model_id: ID of the model to use (e.g., "anthropic/claude-3-haiku")
            system_prompt: Optional system prompt to provide context
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (default: 300 to avoid credit limits)
            stream: Whether to stream the response
            
        Returns:
            Response data from the API
        """
        response = None
        try:
            # Prepare the request payload
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,  # Always include max_tokens to avoid credit issues
                "stream": stream
            }
            
            # Add optional parameters if provided
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.debug(f"Sending chat completion request to model: {model_id}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Make the API request
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload
            )
            
            # Check for credit limit issues (402 Payment Required)
            if response.status_code == 402:
                logger.error("Credit limit exceeded error from OpenRouter API")
                error_message = "Not enough credits. You may need to upgrade your OpenRouter account."
                try:
                    error_json = response.json()
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message = error_json['error']['message']
                except:
                    pass
                return {
                    "error": True,
                    "message": error_message,
                    "code": 402
                }
            
            response.raise_for_status()
            
            # Handle streaming responses differently
            if stream:
                return {"stream": True, "response_object": response}
            
            # Process and return the response for non-streaming requests
            result = response.json()
            
            # Check if the response contains an error field
            if 'error' in result:
                logger.error(f"API returned error: {result['error']}")
                return {
                    "error": True,
                    "message": result.get('error', {}).get('message', 'Unknown API error'),
                    "code": result.get('error', {}).get('code', 400)
                }
            
            logger.info(f"Successfully received response from model: {model_id}")
            return result
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            logger.error(f"HTTP error in chat completion: {str(e)}, Status code: {status_code}")
            error_detail = ""
            
            # Try to extract more error details if available
            try:
                if hasattr(e, 'response') and e.response.text:
                    error_detail = e.response.text
            except:
                pass
                
            return {
                "error": True,
                "message": f"API request failed: {str(e)}",
                "detail": error_detail,
                "code": status_code
            }
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            error_detail = ""
            
            # Try to extract more error details if available
            try:
                if response and response.text:
                    error_detail = response.text
            except:
                pass
                
            return {
                "error": True,
                "message": f"Failed to get completion: {str(e)}",
                "detail": error_detail
            }
    
    def generate_image(self, 
                       prompt: str,
                       model_id: str = "stability/sdxl",
                       n: int = 1,
                       size: str = "1024x1024") -> Dict[str, Any]:
        """
        Generate an image using OpenRouter's image generation API
        
        Args:
            prompt: Text description of the desired image
            model_id: ID of the image model to use
            n: Number of images to generate
            size: Size of the image in format "widthxheight"
            
        Returns:
            Response data from the API
        """
        response = None
        try:
            # Prepare the request payload
            payload = {
                "model": model_id,
                "prompt": prompt,
                "n": n,
                "size": size
            }
            
            logger.debug(f"Sending image generation request to model: {model_id}")
            logger.debug(f"Prompt: {prompt}")
            
            # Make the API request
            response = requests.post(
                self.images_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # Process and return the response
            result = response.json()
            logger.info(f"Successfully received image generation response from model: {model_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            error_detail = ""
            
            # Try to extract more error details if available
            try:
                if response and response.text:
                    error_detail = response.text
            except:
                pass
                
            return {
                "error": True,
                "message": f"Failed to generate image: {str(e)}",
                "detail": error_detail
            }

    def generate_dall_e_image(self, 
                             prompt: str,
                             size: str = "1024x1024") -> Dict[str, Any]:
        """
        Generate an image using DALL-E 3 through OpenRouter's image generation API
        
        Args:
            prompt: Text description of the desired image
            size: Size of the image in format "widthxheight"
            
        Returns:
            Response data with image URL
        """
        response = None
        try:
            # Use OpenRouter's direct image generation endpoint for DALL-E 3
            payload = {
                "model": "openai/dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size
            }
            
            logger.debug(f"Sending DALL-E image generation request")
            logger.debug(f"Prompt: {prompt}")
            
            # Make the API request
            response = requests.post(
                self.images_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # Process and return the response
            result = response.json()
            logger.info(f"Successfully received image generation response from DALL-E 3")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in DALL-E image generation: {str(e)}")
            error_detail = ""
            
            # Try to extract more error details if available
            try:
                if response and response.text:
                    error_detail = response.text
            except:
                pass
                
            return {
                "error": True,
                "message": f"Failed to generate image with DALL-E: {str(e)}",
                "detail": error_detail
            }

# Simple test if run directly
if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        print("No API key found. Please set the OPENROUTER_API_KEY environment variable.")
        exit(1)
    
    # Create client
    client = OpenRouterAPI(api_key)
    
    # List models
    models = client.list_models()
    print(f"Available models: {len(models)}")
    for model in models[:5]:  # Show first 5 models
        print(f"- {model.get('id')}: {model.get('name')}")
    
    # Test chat completion
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Hello, how are you today?"}],
        model_id="anthropic/claude-3-haiku"
    )
    
    print("\nChat completion response:")
    print(json.dumps(response, indent=2))