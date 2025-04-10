"""
AI-Powered Prompt Suggestions

This module provides intelligent prompt suggestions for users.
It uses various contexts and categories to generate helpful prompts.
"""

import random
import logging
import json
import os
from replit_db import ReplitDB
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define categories and their associated prompt templates
PROMPT_CATEGORIES = {
    "creative": [
        "Write a short story about {topic}",
        "Create a poem about {topic} in the style of {style}",
        "Describe a fictional character who is {trait}",
        "Imagine a world where {scenario}",
        "Design a fictional creature that lives in {habitat}"
    ],
    "professional": [
        "Write a professional email requesting {request}",
        "Create a business proposal for {business_idea}",
        "Draft a resume summary for a {job_title} position",
        "Compose a LinkedIn post about {industry_topic}",
        "Write a persuasive pitch for {product_or_service}"
    ],
    "educational": [
        "Explain {concept} as if I'm a beginner",
        "Compare and contrast {topic_a} and {topic_b}",
        "Create a study guide for {subject}",
        "What are the key points to understand about {topic}?",
        "Generate a quiz about {topic} with answers"
    ],
    "problem_solving": [
        "How can I solve this problem: {problem}",
        "What are different approaches to tackle {issue}?",
        "Help me troubleshoot {technical_issue}",
        "What are the pros and cons of {option_a} vs {option_b}?",
        "Generate a step-by-step plan for {goal}"
    ],
    "personal_growth": [
        "Suggest habits that can help me improve {skill}",
        "What are effective ways to overcome {challenge}?",
        "Create a 30-day plan for {personal_goal}",
        "How can I become better at {activity}?",
        "What should I know about getting started with {hobby}?"
    ]
}

# Placeholder values to fill in templates
PLACEHOLDER_VALUES = {
    "topic": ["space exploration", "artificial intelligence", "climate change", 
              "virtual reality", "sustainable energy", "ocean life", 
              "ancient civilizations", "quantum physics", "digital art"],
    "style": ["Shakespeare", "Edgar Allan Poe", "Emily Dickinson", 
              "haiku", "sonnet", "free verse", "Dr. Seuss"],
    "trait": ["highly empathetic", "extremely logical", "mysterious", 
              "from another dimension", "can see the future", "immortal"],
    "scenario": ["humans can teleport", "animals can speak", "time travel is common", 
                 "memories can be transferred", "gravity works differently", 
                 "everyone has a superpower"],
    "habitat": ["deep ocean trenches", "floating islands", "underground caves", 
                "gas giant planets", "inside volcanoes", "arctic ice sheets"],
    "request": ["a project deadline extension", "collaboration opportunity", 
                "information about your services", "feedback on a proposal", 
                "a recommendation letter"],
    "business_idea": ["sustainable packaging solution", "AI-powered assistant", 
                      "subscription box service", "remote team management app", 
                      "virtual event platform"],
    "job_title": ["software engineer", "marketing manager", "data scientist", 
                  "graphic designer", "project manager", "financial analyst"],
    "industry_topic": ["remote work trends", "sustainability initiatives", 
                       "digital transformation", "workplace diversity", 
                       "emerging technologies"],
    "product_or_service": ["productivity app", "consulting service", 
                           "eco-friendly product line", "online course", 
                           "wellness program"],
    "concept": ["machine learning", "blockchain technology", "photosynthesis", 
                "compound interest", "natural selection", "cloud computing"],
    "topic_a": ["renewable energy", "traditional publishing", "remote work", 
                "classical education", "analog photography"],
    "topic_b": ["fossil fuels", "self-publishing", "office-based work", 
                "progressive education", "digital photography"],
    "subject": ["world history", "calculus", "organic chemistry", 
                "macroeconomics", "comparative literature", "astronomy"],
    "problem": ["time management challenges", "balancing work and personal life", 
                "optimizing team communication", "reducing environmental impact", 
                "improving customer retention"],
    "issue": ["project delays", "team conflicts", "budget constraints", 
              "quality control problems", "technology adoption"],
    "technical_issue": ["slow website performance", "application crashes", 
                        "database connectivity problems", "memory leaks", 
                        "authentication failures"],
    "option_a": ["working remotely", "custom development", "subscription model", 
                 "early market entry", "specialization"],
    "option_b": ["working in-office", "using off-the-shelf solutions", "one-time purchase model", 
                 "waiting for market maturity", "diversification"],
    "goal": ["launching a startup", "learning a new language", "implementing a new system", 
             "reducing operational costs", "expanding to new markets"],
    "skill": ["public speaking", "time management", "emotional intelligence", 
              "technical writing", "data analysis", "negotiation"],
    "challenge": ["procrastination", "imposter syndrome", "public speaking anxiety", 
                  "decision fatigue", "information overload"],
    "personal_goal": ["daily meditation", "learning a new language", "starting a side business", 
                      "improving fitness", "reducing screen time"],
    "activity": ["networking", "creative writing", "public speaking", 
                 "financial planning", "strategic thinking"],
    "hobby": ["photography", "gardening", "3D printing", "podcasting", 
              "woodworking", "game development"]
}

# Recent trends and topics to make suggestions more relevant
RECENT_TRENDS = [
    "AI safety and ethics",
    "Climate adaptation strategies",
    "Remote and hybrid work best practices",
    "Digital privacy and security",
    "Mental health in the digital age",
    "Sustainable technology",
    "Blockchain applications beyond cryptocurrency",
    "Edge computing and IoT",
    "Virtual reality workspaces",
    "Personalized medicine"
]

def get_trending_topics():
    """
    Get current trending topics from various sources.
    For now, we'll use a predefined list, but this could be enhanced to fetch real trends.
    """
    # Could be extended to fetch real-time trends from Google Trends API, Twitter, etc.
    return RECENT_TRENDS


def fill_prompt_template(template):
    """
    Fill a prompt template with appropriate values.
    
    Args:
        template (str): Prompt template with placeholders
        
    Returns:
        str: Completed prompt with placeholders filled
    """
    # Find all placeholders in the format {placeholder}
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    filled_template = template
    for placeholder in placeholders:
        # If we have values for this placeholder, pick a random one
        if placeholder in PLACEHOLDER_VALUES:
            value = random.choice(PLACEHOLDER_VALUES[placeholder])
            filled_template = filled_template.replace(f"{{{placeholder}}}", value)
    
    return filled_template


def get_suggestions_by_category(category, count=3):
    """
    Get prompt suggestions from a specific category.
    
    Args:
        category (str): The category to get suggestions from
        count (int): Number of suggestions to return
        
    Returns:
        list: List of prompt suggestions
    """
    if category not in PROMPT_CATEGORIES:
        return []
    
    templates = PROMPT_CATEGORIES[category]
    # Randomly select templates up to the requested count
    selected_templates = random.sample(templates, min(count, len(templates)))
    
    # Fill in the templates with values
    suggestions = [fill_prompt_template(template) for template in selected_templates]
    return suggestions


def get_personalized_suggestions(user_history=None, count=3):
    """
    Get personalized prompt suggestions based on user history.
    
    Args:
        user_history (list): Previous user prompts and interactions
        count (int): Number of suggestions to return
        
    Returns:
        list: List of personalized prompt suggestions
    """
    # For now, we'll use a simplified approach.
    # This could be enhanced with more sophisticated personalization.
    
    # If no history, return random suggestions
    if not user_history:
        # Get suggestions from random categories
        categories = random.sample(list(PROMPT_CATEGORIES.keys()), min(3, len(PROMPT_CATEGORIES)))
        suggestions = []
        for category in categories:
            suggestions.extend(get_suggestions_by_category(category, count=1))
        return suggestions[:count]
    
    # TODO: Implement more sophisticated personalization based on user history
    # For now, just return random suggestions
    all_suggestions = []
    for category in PROMPT_CATEGORIES:
        all_suggestions.extend(get_suggestions_by_category(category, count=1))
    
    # Randomly select from all suggestions
    return random.sample(all_suggestions, min(count, len(all_suggestions)))


def get_topical_suggestions(topic=None, count=3):
    """
    Get suggestions related to a specific topic or current trends.
    
    Args:
        topic (str): Topic to focus suggestions on
        count (int): Number of suggestions to return
        
    Returns:
        list: List of topic-related prompt suggestions
    """
    if not topic:
        # Use trending topics
        trending = get_trending_topics()
        topic = random.choice(trending)
    
    # Create templates focused on the topic
    topic_templates = [
        f"What are the latest developments in {topic}?",
        f"How will {topic} change in the next 5 years?",
        f"What are the ethical implications of {topic}?",
        f"Compare different perspectives on {topic}.",
        f"How does {topic} affect everyday life?",
        f"What innovations are happening in {topic}?",
        f"What should beginners know about {topic}?",
        f"Explain {topic} as if you're teaching a class."
    ]
    
    # Select random templates up to the requested count
    selected_templates = random.sample(topic_templates, min(count, len(topic_templates)))
    return selected_templates


def get_mixed_suggestions(count=5):
    """
    Get a mix of different types of suggestions.
    
    Args:
        count (int): Total number of suggestions to return
        
    Returns:
        list: Mixed list of prompt suggestions
    """
    suggestions = []
    
    # Add some category-based suggestions
    categories = random.sample(list(PROMPT_CATEGORIES.keys()), min(2, len(PROMPT_CATEGORIES)))
    for category in categories:
        suggestions.extend(get_suggestions_by_category(category, count=1))
    
    # Add some topic-based suggestions
    suggestions.extend(get_topical_suggestions(count=1))
    
    # If we still need more, add some random ones
    while len(suggestions) < count:
        category = random.choice(list(PROMPT_CATEGORIES.keys()))
        suggestion = get_suggestions_by_category(category, count=1)[0]
        if suggestion not in suggestions:
            suggestions.append(suggestion)
    
    # Return up to the requested count
    return suggestions[:count]


def cache_suggestions():
    """
    Generate and cache a set of prompt suggestions.
    This can be run periodically to refresh the cache.
    
    Returns:
        dict: The cached suggestions
    """
    cached_data = {
        "timestamp": datetime.now().isoformat(),
        "categories": {},
        "trending": get_topical_suggestions(count=5),
        "mixed": get_mixed_suggestions(count=10)
    }
    
    # Generate suggestions for each category
    for category in PROMPT_CATEGORIES:
        cached_data["categories"][category] = get_suggestions_by_category(category, count=5)
    
    # Store in Replit DB
    try:
        ReplitDB.cache_set("prompt_suggestions", cached_data, ttl=3600*24)  # Cache for 24 hours
        logger.info("Prompt suggestions cached successfully")
    except Exception as e:
        logger.error(f"Failed to cache prompt suggestions: {str(e)}")
    
    # Also save to a file as backup
    try:
        with open('cached_suggestions.json', 'w') as f:
            json.dump(cached_data, f)
    except Exception as e:
        logger.error(f"Failed to save suggestions to file: {str(e)}")
    
    return cached_data


def get_cached_suggestions():
    """
    Retrieve cached suggestions or generate new ones if needed.
    
    Returns:
        dict: Prompt suggestions
    """
    try:
        # Try to get from Replit DB first
        cached = ReplitDB.cache_get("prompt_suggestions")
        if cached:
            logger.info("Retrieved prompt suggestions from cache")
            return cached
    except Exception as e:
        logger.warning(f"Failed to retrieve suggestions from cache: {str(e)}")
    
    # Try to get from file
    try:
        with open('cached_suggestions.json', 'r') as f:
            cached = json.load(f)
            logger.info("Retrieved prompt suggestions from file")
            return cached
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to retrieve suggestions from file: {str(e)}")
    
    # Generate new suggestions
    logger.info("Generating new prompt suggestions")
    return cache_suggestions()


if __name__ == "__main__":
    # Test functionality
    print("Testing prompt suggestions module...")
    
    print("\nCreative Suggestions:")
    for suggestion in get_suggestions_by_category("creative", count=3):
        print(f"- {suggestion}")
    
    print("\nPersonalized Suggestions:")
    for suggestion in get_personalized_suggestions(count=3):
        print(f"- {suggestion}")
    
    print("\nTrending Topic Suggestions:")
    for suggestion in get_topical_suggestions(count=3):
        print(f"- {suggestion}")
    
    print("\nMixed Suggestions:")
    for suggestion in get_mixed_suggestions(count=5):
        print(f"- {suggestion}")
    
    # Cache suggestions
    print("\nCaching suggestions...")
    cache_suggestions()
    print("Done!")