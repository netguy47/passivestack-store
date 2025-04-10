import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from replit import db
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ReplitDB:
    """
    A wrapper for the Replit Database to provide additional functionality
    """
    
    # Prefixes for different data types to organize the database
    PREFERENCES_PREFIX = "prefs_"
    CACHE_PREFIX = "cache_"
    STATS_PREFIX = "stats_"
    USER_PREFIX = "user_"
    
    @staticmethod
    def get_all_keys() -> List[str]:
        """Get all keys in the database"""
        try:
            return list(db.keys())
        except Exception as e:
            logger.error(f"Error getting all keys: {str(e)}")
            return []
    
    @staticmethod
    def get_keys_with_prefix(prefix: str) -> List[str]:
        """Get all keys with a specific prefix"""
        try:
            return [key for key in db.keys() if key.startswith(prefix)]
        except Exception as e:
            logger.error(f"Error getting keys with prefix {prefix}: {str(e)}")
            return []
    
    # ------ Preferences Management ------
    
    @staticmethod
    def set_preference(user_id: str, key: str, value: Any) -> bool:
        """
        Set a user preference in the database
        
        Args:
            user_id: The user identifier
            key: The preference key
            value: The preference value (must be JSON serializable)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_key = f"{ReplitDB.PREFERENCES_PREFIX}{user_id}_{key}"
            db[full_key] = json.dumps(value)
            return True
        except Exception as e:
            logger.error(f"Error setting preference {key} for user {user_id}: {str(e)}")
            return False
    
    @staticmethod
    def get_preference(user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a user preference from the database
        
        Args:
            user_id: The user identifier
            key: The preference key
            default: The default value to return if not found
            
        Returns:
            The preference value, or default if not found
        """
        try:
            full_key = f"{ReplitDB.PREFERENCES_PREFIX}{user_id}_{key}"
            if full_key in db:
                return json.loads(db[full_key])
            return default
        except Exception as e:
            logger.error(f"Error getting preference {key} for user {user_id}: {str(e)}")
            return default
    
    @staticmethod
    def get_all_preferences(user_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a user
        
        Args:
            user_id: The user identifier
            
        Returns:
            dict: All preferences for the user
        """
        try:
            prefix = f"{ReplitDB.PREFERENCES_PREFIX}{user_id}_"
            result = {}
            
            for key in ReplitDB.get_keys_with_prefix(prefix):
                # Extract the preference name without the prefix
                pref_name = key[len(prefix):]
                result[pref_name] = json.loads(db[key])
            
            return result
        except Exception as e:
            logger.error(f"Error getting all preferences for user {user_id}: {str(e)}")
            return {}
    
    # ------ Caching ------
    
    @staticmethod
    def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Cache a value with a TTL (time to live) in seconds
        
        Args:
            key: The cache key
            value: The value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_key = f"{ReplitDB.CACHE_PREFIX}{key}"
            cache_data = {
                "value": value,
                "expires_at": time.time() + ttl
            }
            db[full_key] = json.dumps(cache_data)
            return True
        except Exception as e:
            logger.error(f"Error caching value for key {key}: {str(e)}")
            return False
    
    @staticmethod
    def cache_get(key: str, default: Any = None) -> Any:
        """
        Get a cached value if not expired
        
        Args:
            key: The cache key
            default: Default value to return if not found or expired
            
        Returns:
            The cached value if not expired, otherwise the default value
        """
        try:
            full_key = f"{ReplitDB.CACHE_PREFIX}{key}"
            if full_key in db:
                cache_data = json.loads(db[full_key])
                current_time = time.time()
                
                # Check if the cache is still valid
                if cache_data["expires_at"] > current_time:
                    return cache_data["value"]
                
                # Delete expired cache entry
                del db[full_key]
            
            return default
        except Exception as e:
            logger.error(f"Error getting cached value for key {key}: {str(e)}")
            return default
    
    @staticmethod
    def cache_clear_expired() -> int:
        """
        Clear all expired cache entries
        
        Returns:
            int: Number of cleared entries
        """
        try:
            count = 0
            current_time = time.time()
            
            for key in ReplitDB.get_keys_with_prefix(ReplitDB.CACHE_PREFIX):
                try:
                    cache_data = json.loads(db[key])
                    if cache_data["expires_at"] <= current_time:
                        del db[key]
                        count += 1
                except:
                    # If any error occurs with this entry, delete it
                    del db[key]
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"Error clearing expired cache: {str(e)}")
            return 0
    
    # ------ Usage Statistics ------
    
    @staticmethod
    def increment_stat(category: str, name: str, value: int = 1) -> bool:
        """
        Increment a usage statistic
        
        Args:
            category: Statistic category (e.g., 'api_calls', 'model_usage')
            name: Statistic name (e.g., 'total', 'claude-3')
            value: Amount to increment by (default: 1)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_key = f"{ReplitDB.STATS_PREFIX}{category}_{name}"
            
            # Initialize if not exists
            if full_key not in db:
                db[full_key] = "0"
            
            current = int(db[full_key])
            db[full_key] = str(current + value)
            return True
        except Exception as e:
            logger.error(f"Error incrementing stat {category}_{name}: {str(e)}")
            return False
    
    @staticmethod
    def get_stat(category: str, name: str, default: int = 0) -> int:
        """
        Get a usage statistic
        
        Args:
            category: Statistic category
            name: Statistic name
            default: Default value if not found
            
        Returns:
            int: The statistic value
        """
        try:
            full_key = f"{ReplitDB.STATS_PREFIX}{category}_{name}"
            if full_key in db:
                return int(db[full_key])
            return default
        except Exception as e:
            logger.error(f"Error getting stat {category}_{name}: {str(e)}")
            return default
    
    @staticmethod
    def get_all_stats() -> Dict[str, Dict[str, int]]:
        """
        Get all usage statistics
        
        Returns:
            dict: All statistics organized by category
        """
        try:
            result = {}
            
            for key in ReplitDB.get_keys_with_prefix(ReplitDB.STATS_PREFIX):
                # Extract the category and name from the key
                parts = key[len(ReplitDB.STATS_PREFIX):].split('_', 1)
                if len(parts) != 2:
                    continue
                
                category, name = parts
                
                if category not in result:
                    result[category] = {}
                
                result[category][name] = int(db[key])
            
            return result
        except Exception as e:
            logger.error(f"Error getting all stats: {str(e)}")
            return {}
    
    # ------ User Data Storage ------
    
    @staticmethod
    def store_user_data(user_id: str, data: Dict[str, Any]) -> bool:
        """
        Store user data in the database
        
        Args:
            user_id: The user identifier
            data: User data to store (must be JSON serializable)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_key = f"{ReplitDB.USER_PREFIX}{user_id}"
            db[full_key] = json.dumps(data)
            return True
        except Exception as e:
            logger.error(f"Error storing data for user {user_id}: {str(e)}")
            return False
    
    @staticmethod
    def get_user_data(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data from the database
        
        Args:
            user_id: The user identifier
            
        Returns:
            dict or None: User data if found, None otherwise
        """
        try:
            full_key = f"{ReplitDB.USER_PREFIX}{user_id}"
            if full_key in db:
                return json.loads(db[full_key])
            return None
        except Exception as e:
            logger.error(f"Error getting data for user {user_id}: {str(e)}")
            return None
    
    @staticmethod
    def get_all_users() -> List[str]:
        """
        Get all user IDs in the database
        
        Returns:
            list: All user IDs
        """
        try:
            prefix = ReplitDB.USER_PREFIX
            return [key[len(prefix):] for key in ReplitDB.get_keys_with_prefix(prefix)]
        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return []
    
    # ------ Database Management ------
    
    @staticmethod
    def clear_all(confirm: bool = False) -> bool:
        """
        Clear all data in the database (DANGEROUS!)
        
        Args:
            confirm: Set to True to confirm the action
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not confirm:
            logger.warning("Database clear not confirmed. Set confirm=True to proceed.")
            return False
        
        try:
            for key in db.keys():
                del db[key]
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    @staticmethod
    def clear_prefix(prefix: str, confirm: bool = False) -> int:
        """
        Clear all data with a specific prefix
        
        Args:
            prefix: The key prefix to clear
            confirm: Set to True to confirm the action
            
        Returns:
            int: Number of cleared entries
        """
        if not confirm:
            logger.warning(f"Clear of prefix '{prefix}' not confirmed. Set confirm=True to proceed.")
            return 0
        
        try:
            count = 0
            for key in ReplitDB.get_keys_with_prefix(prefix):
                del db[key]
                count += 1
            return count
        except Exception as e:
            logger.error(f"Error clearing prefix {prefix}: {str(e)}")
            return 0


# Example usage
if __name__ == "__main__":
    # Set a preference
    ReplitDB.set_preference("user123", "theme", "dark")
    ReplitDB.set_preference("user123", "model", "claude-3-haiku")
    
    # Get preferences
    print(ReplitDB.get_preference("user123", "theme"))  # "dark"
    print(ReplitDB.get_all_preferences("user123"))  # {"theme": "dark", "model": "claude-3-haiku"}
    
    # Cache data
    ReplitDB.cache_set("search_results_cats", ["result1", "result2"], ttl=60)  # 60 seconds TTL
    print(ReplitDB.cache_get("search_results_cats"))  # ["result1", "result2"]
    
    # Track usage statistics
    ReplitDB.increment_stat("api_calls", "total")
    ReplitDB.increment_stat("model_usage", "claude-3-haiku")
    print(ReplitDB.get_stat("api_calls", "total"))  # 1
    print(ReplitDB.get_all_stats())  # {"api_calls": {"total": 1}, "model_usage": {"claude-3-haiku": 1}}