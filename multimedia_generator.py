import os
import base64
import json
import logging
import requests
import re
import hashlib
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_text_to_speech(text, language='en', output_file=None):
    """
    Convert text to speech and save as an MP3 file
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (default: 'en' for English)
        output_file (str, optional): Output file path. If None, a temporary file is created.
        
    Returns:
        str: Path to the generated audio file
    """
    try:
        logger.info(f"Converting text to speech: {text[:30]}...")
        
        # Try to import gTTS dynamically
        try:
            from gtts import gTTS
        except ImportError:
            return {
                "success": False,
                "error": "Text-to-speech functionality is not available. The gTTS library is missing."
            }
        
        # Generate output file path if not provided
        if not output_file:
            # Save to static folder with timestamp
            import time
            timestamp = int(time.time())
            output_file = f"static/audio/tts_{timestamp}.mp3"
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create gTTS object
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to file
        tts.save(output_file)
        logger.info(f"Text-to-speech audio saved to {output_file}")
        
        return {
            "success": True,
            "file_path": output_file,
            "url": f"/{output_file}"  # Relative URL for web access
        }
    
    except Exception as e:
        logger.error(f"Error in text-to-speech generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate speech: {str(e)}"
        }

def generate_text_to_image(prompt, width=512, height=512, output_file=None, use_ai=True):
    """
    Convert text prompt to image using AI models
    
    Args:
        prompt (str): Text description for image generation
        width (int): Width of the generated image
        height (int): Height of the generated image
        output_file (str, optional): Output file path. If None, a temporary file is created.
        use_ai (bool): Whether to use AI models for generation (default: True)
        
    Returns:
        dict: Result with success status and file path or error
    """
    try:
        logger.info(f"Generating image from prompt: {prompt[:30]}...")
        
        # Generate output file path if not provided
        if not output_file:
            # Save to static folder with timestamp
            import time
            timestamp = int(time.time())
            output_file = f"static/images/generated_{timestamp}.png"
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Try to use OpenRouter API for AI-generated images
        if use_ai:
            try:
                from openrouter_api import OpenRouterAPI
                
                # Create OpenRouter API client
                openrouter = OpenRouterAPI()
                
                # Create size parameter
                size = f"{width}x{height}"
                
                # Check if API key is available
                if not openrouter.api_key:
                    logger.warning("No OpenRouter API key available. Falling back to local generation.")
                    raise ValueError("No API key available")
                
                # Try DALL-E first as it tends to be more realistic
                logger.info(f"Attempting to generate image using DALL-E via OpenRouter...")
                
                # Generate image using OpenRouter's DALL-E option
                try:
                    result = openrouter.generate_dall_e_image(prompt, size)
                    
                    # Check for either standard OpenAI format or OpenRouter format
                    # OpenRouter might return different format than direct OpenAI
                    if result and 'data' in result and len(result['data']) > 0:
                        if 'url' in result['data'][0]:
                            image_url = result['data'][0]['url']
                            logger.info(f"Successfully generated image with DALL-E. Downloading from {image_url}...")
                            
                            # Download the image
                            response = requests.get(image_url)
                            response.raise_for_status()
                            
                            # Save the image
                            with open(output_file, 'wb') as f:
                                f.write(response.content)
                            
                            logger.info(f"AI-generated image saved to {output_file}")
                            
                            return {
                                "success": True,
                                "file_path": output_file,
                                "url": f"/{output_file}",
                                "source": "dalle"
                            }
                        elif 'b64_json' in result['data'][0]:
                            # Handle base64 encoded image
                            b64_data = result['data'][0]['b64_json']
                            logger.info(f"Received base64 encoded image from DALL-E")
                            # Decode and save directly
                            image_data = base64.b64decode(b64_data)
                            with open(output_file, 'wb') as f:
                                f.write(image_data)
                            
                            logger.info(f"AI-generated image saved to {output_file}")
                            return {
                                "success": True,
                                "file_path": output_file,
                                "url": f"/{output_file}",
                                "source": "dalle"
                            }
                except Exception as dalle_error:
                    logger.error(f"Error generating image with DALL-E: {str(dalle_error)}")
                    # Fall back to Stability Diffusion
                
                # Try Stability Diffusion as a fallback
                try:
                    logger.info(f"Attempting to generate image using Stability Diffusion via OpenRouter...")
                    result = openrouter.generate_image(
                        prompt=prompt,
                        model_id="stability/sdxl",
                        size=size
                    )
                    
                    # Check for either standard OpenAI format or OpenRouter format
                    if result and 'data' in result and len(result['data']) > 0:
                        if 'url' in result['data'][0]:
                            image_url = result['data'][0]['url']
                            logger.info(f"Successfully generated image with Stability. Downloading from {image_url}...")
                            
                            # Download the image
                            response = requests.get(image_url)
                            response.raise_for_status()
                            
                            # Save the image
                            with open(output_file, 'wb') as f:
                                f.write(response.content)
                            
                            logger.info(f"AI-generated image saved to {output_file}")
                            
                            return {
                                "success": True,
                                "file_path": output_file,
                                "url": f"/{output_file}",
                                "source": "stability"
                            }
                        elif 'b64_json' in result['data'][0]:
                            # Handle base64 encoded image
                            b64_data = result['data'][0]['b64_json']
                            logger.info(f"Received base64 encoded image from Stability")
                            # Decode and save directly
                            image_data = base64.b64decode(b64_data)
                            with open(output_file, 'wb') as f:
                                f.write(image_data)
                            
                            logger.info(f"AI-generated image saved to {output_file}")
                            return {
                                "success": True,
                                "file_path": output_file,
                                "url": f"/{output_file}",
                                "source": "stability"
                            }
                except Exception as sd_error:
                    logger.error(f"Error generating image with Stability Diffusion: {str(sd_error)}")
                    # Fall back to local generation
            
            except Exception as api_error:
                logger.error(f"Error using OpenRouter API for image generation: {str(api_error)}")
                # Continue to fallback method
        
        # Generate a visual representation of the prompt using text-based drawing
        # This is a direct approach that doesn't require external services
        logger.info("Falling back to local abstract image generation...")
        try:
            # Create a blank image
            img = Image.new('RGB', (width, height), color=(33, 33, 33))
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Add a gradient background
            for y in range(height):
                r = int(33 + (200 * y / height))
                g = int(33 + (160 * y / height))
                b = int(73 + (100 * y / height))
                for x in range(width):
                    draw.point((x, y), fill=(r, g, b))
            
            # Add a title
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            draw.rectangle(((10, 10), (width-10, 60)), fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((20, 20), f"AI Image: {title}", fill=(255, 255, 255), font=font)
            
            # Draw some abstract shapes based on the hash of the prompt
            # This creates a unique pattern for each prompt
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            # Use the hash to determine colors and positions
            hash_values = [int(prompt_hash[i:i+2], 16) for i in range(0, 32, 2)]
            
            # Make sure we have enough hash values for all the operations
            # Duplicate or extend the hash_values if needed
            while len(hash_values) < 16:
                hash_values.extend(hash_values)
            
            # Draw some circles with bounds checking
            for i in range(min(8, len(hash_values)-5)):  # Make sure we have enough values for the offsets
                # Ensure values are within bounds and account for padding
                safe_width = max(100, width)
                safe_height = max(150, height)
                
                circle_x = 50 + (hash_values[i] % (safe_width - 100))
                circle_y = 100 + (hash_values[i+1] % (safe_height - 150))
                circle_size = 20 + (hash_values[i+2] % 60)  # Reduced max size to avoid going outside bounds
                
                # Make sure the circle fits within the image
                circle_size = min(circle_size, min(circle_x, safe_width - circle_x, circle_y, safe_height - circle_y) - 5)
                
                circle_color = (hash_values[i+3] % 255, hash_values[i+4] % 255, hash_values[i+5] % 255)
                
                # Only draw if the circle will be visible and within bounds
                if circle_size > 0:
                    draw.ellipse(
                        [(circle_x - circle_size, circle_y - circle_size), 
                         (circle_x + circle_size, circle_y + circle_size)], 
                        fill=circle_color
                    )
            
            # Draw some lines with bounds checking
            for i in range(min(6, len(hash_values)-15)):  # Make sure we have enough values for all offsets
                line_start_x = hash_values[i+8] % width
                line_start_y = hash_values[i+9] % height
                line_end_x = hash_values[i+10] % width
                line_end_y = hash_values[i+11] % height
                line_color = (hash_values[i+12] % 255, hash_values[i+13] % 255, hash_values[i+14] % 255)
                line_width = 1 + (hash_values[i+15] % 5)
                
                # Draw the line
                draw.line([(line_start_x, line_start_y), (line_end_x, line_end_y)], fill=line_color, width=line_width)
            
            # Add the prompt text at the bottom
            draw.rectangle(((10, height-60), (width-10, height-10)), fill=(0, 0, 0), outline=(255, 255, 255))
            
            # Wrap text to fit the width
            text_y = height - 50
            words = prompt.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                # In newer Pillow versions, we use font.getbbox() instead of getsize()
                try:
                    # First try getbbox (Pillow >= 9.2.0)
                    text_width = font.getbbox(test_line)[2]
                except (AttributeError, TypeError):
                    try:
                        # Then try getsize (Pillow < 9.2.0)
                        text_width = font.getsize(test_line)[0]
                    except (AttributeError, TypeError):
                        # Estimate text width if font methods unavailable
                        text_width = len(test_line) * 8
                
                if text_width < width - 40:
                    line = test_line
                else:
                    draw.text((20, text_y), line, fill=(255, 255, 255), font=font)
                    text_y += 20
                    line = word + " "
                    
                    # Break if we're going off the image
                    if text_y > height - 20:
                        break
            
            # Draw the last line
            if line:
                draw.text((20, text_y), line, fill=(255, 255, 255), font=font)
            
            # Save the image
            img.save(output_file)
            
            logger.info(f"Generated abstract image saved to {output_file}")
            
            return {
                "success": True,
                "file_path": output_file,
                "url": f"/{output_file}",
                "source": "local_abstract"
            }
        
        except Exception as art_error:
            logger.error(f"Error generating abstract image: {str(art_error)}")
            try:
                # As a fallback, create a very simple image
                img = Image.new('RGB', (width, height), color=(33, 33, 33))
                draw = ImageDraw.Draw(img)
                
                # Add some text
                font = ImageFont.load_default()
                draw.text((20, 20), "AI Image Generation", fill=(255, 255, 255), font=font)
                draw.text((20, 60), f"Prompt: {prompt[:50]}", fill=(200, 200, 200), font=font)
                
                # Save the image
                img.save(output_file)
                
                logger.info(f"Generated simple image saved to {output_file}")
                
                return {
                    "success": True,
                    "file_path": output_file,
                    "url": f"/{output_file}",
                    "source": "local_simple"
                }
            except Exception as simple_error:
                logger.error(f"Error generating simple image: {str(simple_error)}")
                return {
                    "success": False,
                    "error": f"Failed to generate image: {str(simple_error)}"
                }
    
    except Exception as e:
        logger.error(f"Error in text-to-image generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate image: {str(e)}"
        }

def generate_text_to_video(text, duration=5, output_file=None):
    """
    Create a simple animated GIF to serve as a video using PIL
    
    Args:
        text (str): Text to display in the video
        duration (int): Duration of the video in seconds (approximated in the GIF)
        output_file (str, optional): Output file path. If None, a temporary file is created.
        
    Returns:
        dict: Result with success status and file path or error
    """
    try:
        logger.info(f"Generating video from text: {text[:30]}...")
        
        # Since MoviePy is having issues, we'll create an animated GIF instead
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageSequence
        except ImportError:
            return {
                "success": False,
                "error": "Required libraries for video generation are missing."
            }
            
        # Generate output file path if not provided
        if not output_file:
            # Save to static folder with timestamp
            import time
            timestamp = int(time.time())
            output_file = f"static/videos/generated_{timestamp}.gif"  # Using GIF instead of MP4
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create a sequence of frames for the animation
        frames = []
        width, height = 480, 270  # 480p size
        
        # Split text into chunks of 30 characters or less
        text_chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            if len(test_chunk) <= 30:
                current_chunk = test_chunk
            else:
                text_chunks.append(current_chunk)
                current_chunk = word
                
        if current_chunk:
            text_chunks.append(current_chunk)
        
        # If no chunks, just use the original text
        if not text_chunks:
            text_chunks = [text]
        
        # Generate a color gradient for each frame
        colors = []
        num_frames = 10  # Number of frames in the animation
        
        for i in range(num_frames):
            # Create color gradient from blue to purple
            r = int(33 + (160 * i / num_frames))  # Red increases
            g = int(33 + (80 * i / num_frames))   # Green increases somewhat
            b = int(173 - (50 * i / num_frames))  # Blue decreases
            colors.append((r, g, b))
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Create each frame
        for i in range(num_frames):
            img = Image.new('RGB', (width, height), color=(33, 33, 33))
            draw = ImageDraw.Draw(img)
            
            # Add a gradient background
            for y in range(height):
                r = int(33 + (colors[i][0] * y / height))
                g = int(33 + (colors[i][1] * y / height))
                b = int(33 + (colors[i][2] * y / height))
                for x in range(width):
                    draw.point((x, y), fill=(r, g, b))
            
            # Add a title
            title = "AI-Generated Video"
            draw.rectangle(((10, 10), (width-10, 50)), fill=(0, 0, 0, 128), outline=(255, 255, 255))
            draw.text((20, 15), title, fill=(255, 255, 255), font=font)
            
            # Add text chunks with animation (different chunks visible in different frames)
            chunk_index = i % min(len(text_chunks), 3)  # Cycle through up to 3 chunks
            text_y = height // 2 - 30
            
            # Show current chunk and next chunks if available
            for j in range(3):  # Show up to 3 chunks at once
                idx = (chunk_index + j) % len(text_chunks)
                chunk = text_chunks[idx]
                
                # Different styles for current vs. next chunks
                if j == 0:  # Current chunk (highlighted)
                    text_color = (255, 255, 255)
                    font_to_use = font
                else:  # Next chunks (dimmed)
                    text_color = (200, 200, 200)
                    font_to_use = small_font
                
                # Center the text
                try:
                    # First try getbbox (Pillow >= 9.2.0)
                    text_width = font_to_use.getbbox(chunk)[2]
                except (AttributeError, TypeError):
                    try:
                        # Then try getsize (Pillow < 9.2.0)
                        text_width = font_to_use.getsize(chunk)[0]
                    except (AttributeError, TypeError):
                        # Estimate text width if font methods unavailable
                        text_width = len(chunk) * 8
                
                text_x = (width - text_width) // 2
                draw.text((text_x, text_y + j*40), chunk, fill=text_color, font=font_to_use)
            
            # Add a decorative element that changes with each frame
            circle_size = 20 + (i * 3) % 40
            circle_x = width // 2
            circle_y = height - 50
            circle_color = colors[(i + 5) % num_frames]  # Offset the color for contrast
            
            draw.ellipse(
                [(circle_x - circle_size, circle_y - circle_size), 
                 (circle_x + circle_size, circle_y + circle_size)], 
                fill=circle_color
            )
            
            frames.append(img)
        
        # Save as animated GIF
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration * 100,  # Duration in ms per frame
            loop=0  # Loop forever
        )
        
        logger.info(f"Generated animated GIF saved to {output_file}")
        
        # Also generate audio for the text
        audio_file = None
        tts_result = generate_text_to_speech(text)
        
        if tts_result["success"]:
            audio_file = tts_result["file_path"]
            logger.info(f"Generated audio saved to {audio_file}")
        
        return {
            "success": True,
            "file_path": output_file,
            "url": f"/{output_file}",  # Relative URL for web access
            "audio_url": f"/{audio_file}" if audio_file else None  # Include audio URL if available
        }
    
    except Exception as e:
        logger.error(f"Error in text-to-video generation: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate video: {str(e)}"
        }

def image_to_base64(image_path):
    """
    Convert an image file to a base64-encoded string
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64-encoded image string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None