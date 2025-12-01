"""
Script to test the entire system
Usage: python test_system.py --test_image ./test.png
"""

import argparse
import requests
from pathlib import Path
import json

def test_health(base_url: str):
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_stats(base_url: str, api_key: str):
    """Test stats endpoint"""
    print("\n" + "="*60)
    print("Testing Stats Endpoint")
    print("="*60)
    
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{base_url}/stats", headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_search(base_url: str, api_key: str, image_path: str, top_k: int = 5):
    """Test search endpoint"""
    print("\n" + "="*60)
    print("Testing Search Endpoint")
    print("="*60)
    
    headers = {"X-API-Key": api_key}
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {'top_k': top_k}
        response = requests.post(
            f"{base_url}/search",
            headers=headers,
            files=files,
            params=params
        )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Query Time: {data['query_time_ms']:.2f}ms")
        print(f"Total Results: {data['total_results']}")
        print("\nTop Results:")
        for i, result in enumerate(data['results'][:5], 1):
            print(f"\n{i}. Trademark ID: {result['trademark_id']}")
            print(f"   Similarity: {result['similarity_score']:.4f}")
            print(f"   Name: {result['metadata'].get('name', 'N/A')}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def main():
    parser = argparse.ArgumentParser(description="Test trademark system")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000",
                       help="API base URL")
    parser.add_argument("--api_key", type=str, default="your-secret-api-key-here",
                       help="API key")
    parser.add_argument("--test_image", type=str, required=True,
                       help="Path to test image")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of results to return")
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRADEMARK SYSTEM TEST SUITE")
    print("="*60)
    print(f"Base URL: {args.base_url}")
    print(f"Test Image: {args.test_image}")
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_health(args.base_url)))
    results.append(("Database Stats", test_stats(args.base_url, args.api_key)))
    results.append(("Search", test_search(args.base_url, args.api_key, args.test_image, args.top_k)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

if __name__ == "__main__":
    main()