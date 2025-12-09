"""
Test Script: Dynamic BERT Router

Quick test to verify dynamic learning works correctly.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmm_optimizer.routing.dynamic_bert_router import DynamicBERTRouter
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory, Performance


def create_mock_state(iteration: int, mape: float) -> MMMState:
    """Create mock state for testing."""
    return MMMState(
        iteration=iteration,
        performance=Performance(mape=mape, r2=0.85, mae=1000),
        model_config={
            "hyperparameters": {
                "tv_adstock_decay": 0.5,
                "digital_adstock_decay": 0.4
            }
        },
        business_context={"industry": "CPG"},
        convergence_score=0.0
    )


def test_dynamic_router():
    """Test dynamic router functionality."""
    print("=" * 70)
    print("Testing Dynamic BERT Router")
    print("=" * 70)
    
    # Initialize router
    print("\n1. Initializing router...")
    try:
        router = DynamicBERTRouter(
            pretrained_model_path="./mmm_optimizer/bert_router_trained",
            training_data_path="./data/test_training.json",
            model_save_path="./mmm_optimizer/bert_router_trained_updated",
            update_interval=1,  # Update after each iteration for testing
            batch_update_size=3,  # Update when 3 examples accumulated
            save_after_update=True  # Save model after each update
        )
        print("   ✓ Router initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Test routing
    print("\n2. Testing routing...")
    history = MMMHistory()
    
    for i in range(1, 11):
        state = create_mock_state(iteration=i, mape=20.0 - i)
        
        try:
            agent, confidence = router.route(state, history, return_confidence=True)
            print(f"   Iteration {i}: {agent} (confidence: {confidence:.2%})")
            
            # Record iteration
            outcome = {"mape": 20.0 - i - 0.5, "r2": 0.85 + i*0.01}
            router.record_iteration(state, history, agent, outcome)
            
        except Exception as e:
            print(f"   ✗ Routing failed at iteration {i}: {e}")
            return False
    
    print("   ✓ Routing test passed")
    
    # Check statistics
    print("\n3. Checking statistics...")
    stats = router.get_statistics()
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Update count: {stats['update_count']}")
    print(f"   Pending examples: {stats['pending_examples']}")
    print(f"   Label distribution: {stats['label_distribution']}")
    
    if stats['total_examples'] != 10:
        print(f"   ✗ Expected 10 examples, got {stats['total_examples']}")
        return False
    
    if stats['update_count'] < 1:
        print(f"   ✗ Expected at least 1 update, got {stats['update_count']}")
        return False
    
    print(f"   Model updates performed: {stats['update_count']}")
    print(f"   Pending examples: {stats['pending_examples']}")
    
    print("   ✓ Statistics check passed")
    
    # Test data export
    print("\n4. Testing data export...")
    try:
        router.export_training_data("./data/test_export.json")
        print("   ✓ Data exported successfully")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False
    
    # Check if model was saved
    print("\n5. Verifying model persistence...")
    from pathlib import Path
    model_path = Path("./mmm_optimizer/bert_router_trained_updated")
    if model_path.exists():
        metadata_file = model_path / "update_metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"   ✓ Model saved with {metadata['update_count']} updates")
        else:
            print("   ✓ Model directory exists")
    else:
        print("   ✗ Model was not saved")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Online Learning Verified")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - {stats['total_examples']} training examples collected")
    print(f"  - {stats['update_count']} online model updates performed")
    print(f"  - Model saved and ready for continued learning")
    return True


if __name__ == "__main__":
    success = test_dynamic_router()
    sys.exit(0 if success else 1)
