import os
import time
from train_rag_model import main, get_rag_manager

def test_cache_system():
    """Test the caching system"""
    print("🔥 Testing Cache System")
    print("=" * 40)
    
    # Get RAG manager
    rag_manager = get_rag_manager()
    cache_file = rag_manager.cache_file
    
    # Test 1: First run (should create cache)
    print("\n1️⃣ First run (cold start - should create cache):")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"   Removed existing cache: {cache_file}")
    
    start_time = time.time()
    response1 = main("What is FUTA?")
    first_run_time = time.time() - start_time
    
    print(f"   ✅ First run completed in {first_run_time:.2f} seconds")
    print(f"   Cache file created: {os.path.exists(cache_file)}")
    print(f"   Response length: {len(response1)} characters")
    
    # Test 2: Second run (should use cache)
    print("\n2️⃣ Second run (warm start - should use cache):")
    start_time = time.time()
    response2 = main("What faculties are in FUTA?")
    second_run_time = time.time() - start_time
    
    print(f"   ✅ Second run completed in {second_run_time:.2f} seconds")
    print(f"   Cache file exists: {os.path.exists(cache_file)}")
    print(f"   Response length: {len(response2)} characters")
    
    # Test 3: Performance comparison
    print(f"\n📊 Performance Comparison:")
    print(f"   First run (cold):  {first_run_time:.2f}s")
    print(f"   Second run (warm): {second_run_time:.2f}s")
    
    if second_run_time < first_run_time:
        speedup = first_run_time / second_run_time
        print(f"   🚀 Speedup: {speedup:.1f}x faster with cache!")
    
    # Test 4: Cache rebuild
    print("\n3️⃣ Testing cache rebuild:")
    start_time = time.time()
    response3 = main("Tell me about student life in FUTA", force_rebuild=True)
    rebuild_time = time.time() - start_time
    
    print(f"   ✅ Cache rebuild completed in {rebuild_time:.2f} seconds")
    
    print("\n🎉 Cache testing completed!")

if __name__ == "__main__":
    test_cache_system()