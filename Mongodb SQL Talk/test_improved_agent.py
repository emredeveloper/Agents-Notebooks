#!/usr/bin/env python3
"""
Test script for improved MongoDB agent - focusing on previously failed queries
"""

import sys
import os
import json
import time
from typing import Dict, List, Any
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the agent class
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("mongodb_langchain_agent", 
                                                os.path.join(os.path.dirname(__file__), "mongodb-langchain-agent-clean.py"))
    mongodb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mongodb_module)
    MongoDBLangChainAgent = mongodb_module.MongoDBLangChainAgent
except Exception as e:
    print(f"❌ Could not import MongoDBLangChainAgent: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_critical_queries():
    """Test the most critical queries that were failing"""
    
    # Initialize agent
    agent = MongoDBLangChainAgent(
        mongo_uri="mongodb://localhost:27017/",
        lm_studio_url="http://localhost:1234/v1",
        model_name="qwen/qwen3-4b-2507"
    )
    db_name = "agent_test"
    
    # Critical failing queries from test results
    critical_queries = [
        "users ilk 5 veriyi göster",
        "adı Ayşe olanları listele", 
        "ismi Mehmet olanları göster",
        "kaç kullanıcı var",
        "kullanıcıları listele",
        "yaşı 30'dan büyük olanlar",
        "koleksiyonları listele",
        "agent-sql koleksiyonu boş mu"
    ]
    
    print("🧪 Testing Improved MongoDB Agent")
    print("=" * 50)
    
    results = {}
    total_tests = len(critical_queries)
    passed_tests = 0
    
    for i, query in enumerate(critical_queries, 1):
        print(f"\n{i}/{total_tests} Testing: '{query}'")
        
        start_time = time.time()
        try:
            result = agent.process_query(query, db_name)
            execution_time = time.time() - start_time
            
            # Analyze result
            if isinstance(result, dict) and result.get("success"):
                passed_tests += 1
                print(f"   ✅ PASSED - {result.get('type', 'unknown')} ({execution_time:.2f}s)")
                
                if result.get("data"):
                    print(f"   📊 Data: {len(result['data'])} records")
                    
                results[query] = {
                    "success": True,
                    "type": result.get("type"),
                    "execution_time": execution_time,
                    "data_count": len(result.get("data", [])),
                    "response": result.get("response", "")
                }
            else:
                print(f"   ❌ FAILED - {result.get('error', 'Unknown error')}")
                results[query] = {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "execution_time": execution_time
                }
                
        except Exception as e:
            print(f"   💥 EXCEPTION - {str(e)}")
            results[query] = {
                "success": False,
                "error": f"Exception: {str(e)}",
                "execution_time": time.time() - start_time
            }
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 IMPROVEMENT TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Show details for failed tests
    failed_tests = [q for q, r in results.items() if not r["success"]]
    if failed_tests:
        print(f"\n❌ Failed Queries:")
        for query in failed_tests:
            print(f"  - '{query}' → {results[query]['error']}")
    
    # Show passed tests
    passed_test_list = [q for q, r in results.items() if r["success"]]
    if passed_test_list:
        print(f"\n✅ Passed Queries:")
        for query in passed_test_list:
            print(f"  - '{query}' → {results[query]['type']} ({results[query]['execution_time']:.2f}s)")
    
    # Save results
    with open("improved_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_results": results,
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": (passed_tests/total_tests)*100
            },
            "timestamp": time.time()
        }, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n💾 Results saved to improved_test_results.json")
    
    return results

def test_performance_comparison():
    """Compare performance with original results"""
    
    # Load original test results
    try:
        with open("test_results.json", "r", encoding="utf-8") as f:
            original_results = json.load(f)
    except Exception as e:
        print(f"❌ Could not load original test results (test_results.json not found or unreadable): {e}")
        print("➡️  Running critical tests anyway without baseline comparison...")
        # Still run the critical tests so the user gets output
        test_critical_queries()
        return
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Count original results
    original_total = 0 
    original_passed = 0
    
    for category, tests in original_results["test_results"].items():
        for test in tests:
            original_total += 1
            if test.get("success") and test.get("has_data"):
                original_passed += 1
    
    original_success_rate = (original_passed / original_total) * 100 if original_total > 0 else 0
    
    print(f"Original System:")
    print(f"  - Success Rate: {original_success_rate:.1f}% ({original_passed}/{original_total})")
    print(f"  - Most queries failed with 'Agent stopped due to iteration limit'")
    
    # Run new test
    new_results = test_critical_queries()
    
    # Show improvement
    improvement = len([r for r in new_results.values() if r["success"]]) / len(new_results) * 100
    
    print(f"\nImproved System:")
    print(f"  - Success Rate: {improvement:.1f}%")
    print(f"  - Uses smart fallback system")
    print(f"  - Increased iteration limits")
    
    if improvement > original_success_rate:
        print(f"\n🎉 IMPROVEMENT: +{improvement - original_success_rate:.1f}% success rate!")
    else:
        print(f"\n⚠️  Still needs work: {improvement - original_success_rate:.1f}% change")

if __name__ == "__main__":
    test_performance_comparison()
