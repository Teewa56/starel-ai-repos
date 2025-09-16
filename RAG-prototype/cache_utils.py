import os
import pickle
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for RAG system components"""
    
    def __init__(self, cache_dir="cache", cache_version="1.0"):
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, cache_name):
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_name}.pkl")
    
    def generate_content_hash(self, content):
        """Generate hash for content to detect changes"""
        if isinstance(content, list):
            # For document lists
            text_content = ""
            for item in content:
                if isinstance(item, dict):
                    text_content += str(item.get('text', '')) + str(item.get('source', ''))
                else:
                    text_content += str(item)
        else:
            text_content = str(content)
        
        return hashlib.md5(text_content.encode()).hexdigest()
    
    def save_cache(self, cache_name, data, content_hash=None):
        """
        Save data to cache
        
        Args:
            cache_name (str): Name of the cache
            data (dict): Data to cache
            content_hash (str): Hash of the content for validation
        """
        try:
            cache_data = {
                'data': data,
                'content_hash': content_hash,
                'cache_version': self.cache_version,
                'created_at': datetime.now().isoformat(),
                'cache_name': cache_name
            }
            
            cache_path = self._get_cache_path(cache_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cache {cache_name}: {e}")
            return False
    
    def load_cache(self, cache_name, expected_hash=None):
        """
        Load data from cache
        
        Args:
            cache_name (str): Name of the cache
            expected_hash (str): Expected content hash for validation
            
        Returns:
            dict or None: Cached data if valid, None otherwise
        """
        cache_path = self._get_cache_path(cache_name)
        
        if not os.path.exists(cache_path):
            logger.info(f"Cache file not found: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache structure
            if not isinstance(cache_data, dict) or 'data' not in cache_data:
                logger.warning(f"Invalid cache structure: {cache_name}")
                return None
            
            # Check version compatibility
            if cache_data.get('cache_version') != self.cache_version:
                logger.info(f"Cache version mismatch for {cache_name}")
                return None
            
            # Validate content hash if provided
            if expected_hash and cache_data.get('content_hash') != expected_hash:
                logger.info(f"Content hash mismatch for {cache_name}")
                return None
            
            logger.info(f"Cache loaded successfully: {cache_name}")
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"Error loading cache {cache_name}: {e}")
            return None
    
    def cache_exists(self, cache_name):
        """Check if cache file exists"""
        cache_path = self._get_cache_path(cache_name)
        return os.path.exists(cache_path)
    
    def delete_cache(self, cache_name):
        """Delete a specific cache file"""
        cache_path = self._get_cache_path(cache_name)
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cache deleted: {cache_path}")
                return True
            else:
                logger.info(f"Cache file not found: {cache_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting cache {cache_name}: {e}")
            return False
    
    def clear_all_cache(self):
        """Clear all cache files in the cache directory"""
        try:
            if not os.path.exists(self.cache_dir):
                logger.info("No cache directory to clear")
                return True
            
            cleared_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} cache files")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self, cache_name):
        """Get information about a cache file"""
        cache_path = self._get_cache_path(cache_name)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Get file stats
            stats = os.stat(cache_path)
            file_size = stats.st_size
            modified_time = datetime.fromtimestamp(stats.st_mtime).isoformat()
            
            # Try to get cache metadata
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            return {
                'cache_name': cache_name,
                'file_path': cache_path,
                'file_size': file_size,
                'modified_time': modified_time,
                'cache_version': cache_data.get('cache_version'),
                'created_at': cache_data.get('created_at'),
                'content_hash': cache_data.get('content_hash')
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info for {cache_name}: {e}")
            return {
                'cache_name': cache_name,
                'file_path': cache_path,
                'error': str(e)
            }
    
    def list_all_caches(self):
        """List all cache files with their information"""
        if not os.path.exists(self.cache_dir):
            return []
        
        caches = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                cache_name = filename[:-4]  # Remove .pkl extension
                cache_info = self.get_cache_info(cache_name)
                if cache_info:
                    caches.append(cache_info)
        
        return caches

# Convenience functions for backward compatibility
def save_rag_components(embeddings, chunked_docs, embedding_model, filename="rag_cache.pkl"):
    """Legacy function - saves RAG components using old method"""
    try:
        cache_data = {
            'embeddings': embeddings,
            'chunked_docs': chunked_docs,
            'embedding_model': embedding_model,
            'cache_version': '1.0',
            'created_at': datetime.now().isoformat()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"RAG components saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving RAG components: {e}")
        return False

def load_rag_components(filename="rag_cache.pkl"):
    """Legacy function - loads RAG components using old method"""
    if not os.path.exists(filename):
        return None
        
    try:
        with open(filename, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if it has the required keys
        required_keys = ['embeddings', 'chunked_docs', 'embedding_model']
        if all(key in cache_data for key in required_keys):
            logger.info(f"RAG components loaded from {filename}")
            return cache_data
        else:
            logger.warning(f"Invalid cache structure in {filename}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading RAG components: {e}")
        return None