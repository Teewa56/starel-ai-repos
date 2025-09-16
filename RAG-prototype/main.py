from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
from trrain_rag_model import main, rebuild_rag_system, get_rag_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/ask": {"origins": ["https://starel-frontend.vercel.app", "http://localhost:3000"]},
    r"/health": {"origins": "*"},
    r"/rebuild": {"origins": ["https://starel-frontend.vercel.app", "http://localhost:3000"]}
})

# Global variables for system state
system_initialized = False
initialization_error = None

def initialize_system():
    """Initialize RAG system on startup"""
    global system_initialized, initialization_error
    try:
        logger.info("Initializing RAG system on startup...")
        rag_manager = get_rag_manager()
        rag_manager.get_rag_system()
        system_initialized = True
        initialization_error = None
        logger.info("RAG system initialized successfully")
    except Exception as e:
        system_initialized = False
        initialization_error = str(e)
        logger.error(f"Failed to initialize RAG system: {e}")

# Initialize system on startup
initialize_system()

@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if system_initialized else "unhealthy",
        "initialized": system_initialized,
        "error": initialization_error,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/ask", methods=['POST'])
def ask():
    """Main endpoint for asking questions"""
    try:
        # Check if system is initialized
        if not system_initialized:
            return jsonify({
                "error": "RAG system not initialized",
                "details": initialization_error
            }), 503
        
        # Validate request
        if not request.json:
            return jsonify({"error": "Request must contain JSON data"}), 400
        
        user_prompt = request.json.get("prompt", "").strip()
        if not user_prompt:
            return jsonify({"error": "User prompt not specified or empty"}), 400
        
        # Log the request
        logger.info(f"Received query: {user_prompt[:100]}...")
        
        # Generate response
        response = main(user_prompt)
        
        # Check if response indicates an error
        if response.startswith("Error:"):
            logger.error(f"RAG system error: {response}")
            return jsonify({
                "error": "Internal processing error",
                "details": response
            }), 500
        
        logger.info("Response generated successfully")
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "error": "Internal server error",
            "details": error_msg
        }), 500

@app.route("/rebuild", methods=['POST'])
def rebuild():
    """Endpoint to rebuild RAG system"""
    try:
        logger.info("Rebuilding RAG system...")
        rebuild_rag_system()
        
        # Re-initialize system state
        global system_initialized, initialization_error
        system_initialized = True
        initialization_error = None
        
        logger.info("RAG system rebuilt successfully")
        return jsonify({
            "message": "RAG system rebuilt successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Error rebuilding system: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "error": "Failed to rebuild system",
            "details": error_msg
        }), 500

@app.route("/status", methods=['GET'])
def status():
    """Get detailed system status"""
    try:
        rag_manager = get_rag_manager()
        cache_exists = os.path.exists(rag_manager.cache_file)
        
        return jsonify({
            "system_initialized": system_initialized,
            "cache_exists": cache_exists,
            "cache_file": rag_manager.cache_file,
            "data_directory": rag_manager.data_directory,
            "initialization_error": initialization_error,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error getting status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/ask", "/health", "/rebuild", "/status"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"System initialized: {system_initialized}")
    
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=port,
        threaded=True 
    )