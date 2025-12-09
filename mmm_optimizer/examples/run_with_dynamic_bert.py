"""
Example: Running MMM Optimization with Dynamic BERT Router

This script demonstrates how to use the Dynamic BERT Router for online learning.

The router:
1. Starts with pre-trained model from synthetic data
2. Collects real training examples during optimization
3. Automatically retrains every N iterations with real data
4. Continuously improves its routing decisions

Usage:
    python -m mmm_optimizer.examples.run_with_dynamic_bert
"""

import os
from mmm_optimizer.orchestrator.orchestrator import Orchestrator
from mmm_optimizer.orchestrator.state import MMMState, Performance
from mmm_optimizer.routing.dynamic_bert_router import DynamicBERTRouter


def create_baseline_state() -> MMMState:
    """Create initial MMM state with baseline performance."""
    return MMMState(
        iteration=0,
        performance=Performance(
            mape=18.5,  # High initial error
            r2=0.78,     # Moderate fit
            mae=1850.0
        ),
        model_config={
            "hyperparameters": {
                "tv_adstock_decay": 0.5,
                "digital_adstock_decay": 0.4,
                "print_adstock_decay": 0.3,
                "ooh_adstock_decay": 0.45,
                "tv_saturation_alpha": 2.0,
                "digital_saturation_alpha": 2.2,
                "print_saturation_alpha": 1.8,
                "ooh_saturation_alpha": 2.1,
                "learning_rate": 0.001,
                "regularization": 0.01,
                "batch_size": 32,
                "epochs": 200
            },
            "methodology": {
                "adstock_type": "beta_gamma",
                "saturation_type": "hill",
                "hierarchy_levels": ["brand", "region", "channel"],
                "architecture_type": "hierarchical_neural_additive",
                "loss_type": "mse"
            }
        },
        business_context={
            "industry": "CPG",
            "campaign_type": "Brand Awareness",
            "budget_usd": 5000000,
            "target_roi": 3.5
        },
        convergence_score=0.0
    )


def main():
    """Run optimization session with dynamic BERT router."""
    
    print("=" * 70)
    print("MMM Optimization with Dynamic BERT Router")
    print("=" * 70)
    print()
    
    # Check if ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Create baseline state
    baseline = create_baseline_state()
    
    print(f"Baseline Configuration:")
    print(f"  • MAPE: {baseline.performance.mape:.2f}%")
    print(f"  • R²: {baseline.performance.r2:.4f}")
    print(f"  • Industry: {baseline.business_context['industry']}")
    print(f"  • Budget: ${baseline.business_context['budget_usd']:,}")
    print()
    
    # Initialize orchestrator with dynamic learning
    orchestrator = Orchestrator(
        baseline_state=baseline,
        max_iterations=20,
        convergence_threshold=0.85,
        use_dynamic_learning=True,  # Enable online learning
        retrain_interval=10  # Retrain every 10 iterations
    )
    
    print("\n" + "=" * 70)
    print("Starting Optimization Session")
    print("=" * 70)
    print()
    
    # Run optimization
    results = orchestrator.run_session()
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nSession Status: {results['session_status']}")
    print(f"Iterations Completed: {results['iterations_completed']}")
    print(f"Termination Reason: {results['termination_reason']}")
    print(f"\nFinal Performance:")
    print(f"  • MAPE: {results['final_state'].performance.mape:.2f}%")
    print(f"  • R²: {results['final_state'].performance.r2:.4f}")
    print(f"  • Convergence Score: {results['convergence_score']:.2f}")
    print(f"\nStakeholder Decision: {results['stakeholder_decision']}")
    
    # Get dynamic router statistics
    if orchestrator.use_dynamic_learning:
        stats = orchestrator.dynamic_router.get_statistics()
        print("\n" + "=" * 70)
        print("DYNAMIC BERT ROUTER STATISTICS")
        print("=" * 70)
        print(f"Total Training Examples Collected: {stats['total_examples']}")
        print(f"Number of Retraining Cycles: {stats['retrain_count']}")
        print(f"Agent Distribution:")
        for agent, count in stats['label_distribution'].items():
            print(f"  • {agent}: {count}")
        print(f"\nNext Retrain In: {stats['next_retrain_in']} iterations")
        
        # Export training data
        export_path = "./data/bert_training_export.json"
        orchestrator.dynamic_router.export_training_data(export_path)
        print(f"\n✓ Training data exported to: {export_path}")
    
    print("\n" + "=" * 70)
    print()
    
    # Print optimization history summary
    print("OPTIMIZATION HISTORY:")
    print("-" * 70)
    for i, record in enumerate(results['history'].iterations, 1):
        mape_before = record.state_before.performance.mape
        mape_after = record.state_after.performance.mape
        improvement = mape_before - mape_after
        
        print(f"Iteration {i}:")
        print(f"  Agent: {record.agent_name}")
        print(f"  MAPE: {mape_before:.2f}% → {mape_after:.2f}% (Δ {improvement:+.2f}%)")
        print(f"  Decision: {record.agent_output_structured.decision if record.agent_output_structured else 'N/A'}")
        print()


if __name__ == "__main__":
    main()
