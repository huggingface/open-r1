import subprocess
import threading
import time
from typing import List, Optional

# Import the factory function to get the provider
from src.open_r1.utils.code_providers import get_provider

def test_provider(provider_type="morph", use_router=False, num_samples=2):
    """Test a code execution provider with sample code."""
    print(f"\n{'='*80}\nTesting {provider_type} provider {'with router' if use_router else 'direct'}\n{'='*80}")
    
    # Router URLs - adjust these if you're running the routers on different ports/hosts
    router_config = {}
    if use_router:
        if provider_type == "morph":
            print("Starting local MorphCloud router...")
            # Start router in a separate process
            import subprocess
            import threading
            
            def run_morph_router():
                subprocess.run(["python", "scripts/morph_router.py", "--port", "8001"])
            
            # Start router in a separate thread
            router_thread = threading.Thread(target=run_morph_router)
            router_thread.daemon = True
            router_thread.start()
            time.sleep(3)  # Give the router time to start
            
            router_config = {"morph_router_url": "localhost:8001"}
            print("MorphCloud router started at localhost:8001")
    
    # Create provider
    start_time = time.time()
    provider = get_provider(provider_type=provider_type, num_parallel=2, **router_config)
    
    # Test Python code samples
    python_samples = [
        """
def solution(n):
    # Calculate sum of first n natural numbers
    return n * (n + 1) // 2

# Test with n=10
result = solution(10)
print(result)
""",
        """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Test with n=17
result = 1.0 if is_prime(17) else 0.0
print(result)
"""
    ]
    
    # Use a subset of samples for testing
    test_samples = python_samples[:num_samples]
    
    # Execute code samples
    print(f"Executing {len(test_samples)} Python code samples...")
    results = provider.execute_scripts(test_samples, language="python")
    
    # Print results
    for i, result in enumerate(results):
        print(f"Sample {i+1} reward: {result}")
    
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test code execution providers')
    parser.add_argument('--provider', type=str, choices=['e2b', 'morph', 'local'], default='morph',
                        help='Provider to test (e2b, morph, or local)')
    parser.add_argument('--router', action='store_true', help='Use router mode')
    
    args = parser.parse_args()
    
    test_provider(args.provider, args.router)