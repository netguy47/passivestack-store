import os
import logging
import json
import requests
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ServiceConnector:
    """
    A utility class to connect to various external services
    This serves as a central place to manage API keys and connections
    """
    
    @staticmethod
    def get_secret(key_name: str, fallback: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret from environment variables
        
        Args:
            key_name: Name of the environment variable
            fallback: Optional fallback value
            
        Returns:
            The secret value or fallback
        """
        value = os.environ.get(key_name)
        if value:
            logger.info(f"Found secret for {key_name}")
            return value
        
        logger.warning(f"Secret for {key_name} not found")
        return fallback
    
    @staticmethod
    def validate_service_connection(service_name: str) -> Dict[str, Any]:
        """
        Check if connection to a service is possible
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            Dictionary with status information
        """
        # Define mappings of service names to their API key environment variables
        service_key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'slack': 'SLACK_API_TOKEN',
            'discord': 'DISCORD_BOT_TOKEN',
            'stripe': 'STRIPE_SECRET_KEY',
            'hubspot': 'HUBSPOT_API_KEY'
        }
        
        # Get the appropriate environment variable name
        env_var_name = service_key_mapping.get(service_name.lower())
        if not env_var_name:
            return {
                "service": service_name,
                "status": "error",
                "message": f"Unknown service: {service_name}"
            }
        
        # Check if the API key exists
        key = ServiceConnector.get_secret(env_var_name)
        if not key:
            return {
                "service": service_name,
                "status": "error",
                "message": f"API key not found for {service_name}. Please add {env_var_name} to your environment variables."
            }
        
        # Simple validation (not making actual API calls to avoid rate limits)
        return {
            "service": service_name,
            "status": "available",
            "message": f"API key for {service_name} is available"
        }
    
    @staticmethod
    def check_all_services() -> Dict[str, Dict[str, Any]]:
        """
        Check connection status for all supported services
        
        Returns:
            Dictionary with service statuses
        """
        services = [
            'openai', 'anthropic', 'openrouter', 'google', 'perplexity', 
            'mistral', 'slack', 'discord', 'stripe', 'hubspot'
        ]
        
        results = {}
        for service in services:
            results[service] = ServiceConnector.validate_service_connection(service)
        
        return results

    @staticmethod
    def format_services_status() -> str:
        """
        Generate a formatted report of service connections
        
        Returns:
            Formatted string with service status information
        """
        statuses = ServiceConnector.check_all_services()
        
        # Format as text report
        lines = ["# Service Connection Status", ""]
        
        for service_name, status in statuses.items():
            status_indicator = "✅" if status["status"] == "available" else "❌"
            lines.append(f"{status_indicator} **{service_name.title()}**: {status['message']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_openai_client():
        """Get authenticated OpenAI client if possible"""
        try:
            from openai import OpenAI
            
            api_key = ServiceConnector.get_secret("OPENAI_API_KEY")
            if not api_key:
                return None
            
            return OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    
    @staticmethod
    def get_anthropic_client():
        """Get authenticated Anthropic client if possible"""
        try:
            from anthropic import Anthropic
            
            api_key = ServiceConnector.get_secret("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            
            return Anthropic(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {str(e)}")
            return None
            
    @staticmethod
    def get_openrouter_client():
        """Get authenticated OpenRouter client if possible"""
        try:
            from openrouter_api import OpenRouterAPI
            
            api_key = ServiceConnector.get_secret("OPENROUTER_API_KEY")
            if not api_key:
                return None
            
            return OpenRouterAPI(api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenRouter client: {str(e)}")
            return None

# Add a route in main.py to display the service connection status page
if __name__ == "__main__":
    print(ServiceConnector.format_services_status())