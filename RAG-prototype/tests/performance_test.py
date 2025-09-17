import time
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor

API_BASE = "http://127.0.0.1:5000"

def test_single_request(question):
    """Test a single API request"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/ask",
            json={"prompt": question},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response_time": end_time - start_time,
                "response_length": len(result.get("response", "")),
                "question": question
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text,
                "question": question
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "question": question
        }

def test_response_quality():
    """Test response quality with various questions"""
    print("📊 Testing Response Quality")
    print("=" * 40)
    
    test_questions = [
        "What is FUTA?",
        "When was FUTA established?",
        "What faculties are available in FUTA?",
        "Tell me about FUTA's grading system",
        "What are the research centers in FUTA?",
        "How is student life in FUTA?",
        "What is the admission process for FUTA?",
        "Tell me about FUTA's engineering programs"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        result = test_single_request(question)
        
        if result["success"]:
            print(f"   ✅ Success ({result['response_time']:.2f}s)")
            print(f"   Response: {result['response_length']} chars")
            # Print first 100 chars of response
            if result.get('response_length', 0) > 0:
                print(f"   Preview: [Response received but not shown for brevity]")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

def test_concurrent_requests():
    """Test concurrent requests to simulate multiple users"""
    print("\n🔄 Testing Concurrent Requests")
    print("=" * 40)
    
    questions = [
        "What is FUTA?",
        "What faculties are in FUTA?", 
        "Tell me about student life",
        "What is the grading system?",
        "When was FUTA established?"
    ] * 2  # 10 total requests
    
    print(f"   Sending {len(questions)} concurrent requests...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(test_single_request, questions))
    
    end_time = time.time()
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"   ✅ Successful requests: {len(successful)}/{len(results)}")
    print(f"   ❌ Failed requests: {len(failed)}")
    print(f"   ⏱️  Total time: {end_time - start_time:.2f}s")
    
    if successful:
        avg_response_time = sum(r["response_time"] for r in successful) / len(successful)
        print(f"   📊 Average response time: {avg_response_time:.2f}s")

def test_api_endpoints():
    """Test all API endpoints"""
    print("\n🔗 Testing API Endpoints")
    print("=" * 40)
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/status", None),
        ("POST", "/ask", {"prompt": "What is FUTA?"})
    ]
    
    for method, endpoint, data in endpoints:
        print(f"\n   Testing {method} {endpoint}")
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=30)
            
            print(f"      Status: {response.status_code}")
            if response.status_code == 200:
                print(f"      ✅ Success")
            else:
                print(f"      ❌ Failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")

def main():
    """Run all performance tests"""
    print("🚀 Starting Performance Tests")
    print("📍 Make sure Flask server is running on http://127.0.0.1:5000")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server not running. Start with: python main.py")
        return
    
    # Run tests
    test_api_endpoints()
    test_response_quality()
    test_concurrent_requests()
    
    print("\n🎉 Performance testing completed!")

if __name__ == "__main__":
    main()