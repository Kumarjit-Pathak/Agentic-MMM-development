"""
Run MMM Optimization Session

Entry point script for running a complete multi-agent optimization session.
Loads baseline model, initializes orchestrator, and executes optimization loop.

Usage:
    python scripts/run_session.py \
        --baseline-config path/to/baseline_model.json \
        --business-context path/to/business_context.json \
        --max-iterations 20 \
        --convergence-threshold 0.95

Example baseline_model.json:
{
    "performance": {
        "mape": 15.2,
        "r2": 0.82,
        "mae": 1250.5
    },
    "model_config": {
        "methodology": {
            "adstock": {"type": "geometric"},
            "saturation": {"type": "hill"},
            "hierarchy": {"levels": ["channel", "campaign"]}
        },
        "hyperparameters": {
            "tv_adstock_decay": 0.6,
            "digital_adstock_decay": 0.4,
            "tv_saturation_alpha": 2.0,
            "learning_rate": 0.001
        }
    }
}

Example business_context.json:
{
    "industry": "CPG",
    "campaign_type": "Q4_Holiday_2024",
    "budget": 5000000,
    "channels": ["TV", "Digital", "Print"],
    "key_metrics": ["ROAS", "Carryover", "Saturation"]
}
"""

import argparse
import json
from pathlib import Path
from mmm_optimizer.orchestrator.state import MMMState, Performance
from mmm_optimizer.orchestrator.orchestrator import Orchestrator


def load_baseline_config(path: str) -> dict:
    """Load baseline model configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_business_context(path: str) -> dict:
    """Load business context from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def create_baseline_state(
    baseline_config: dict,
    business_context: dict
) -> MMMState:
    """
    Create initial MMMState from configuration files.
    
    Args:
        baseline_config: Dict with performance + model_config
        business_context: Dict with business metadata
    
    Returns:
        MMMState ready for optimization
    """
    performance = Performance(
        mape=baseline_config["performance"]["mape"],
        r2=baseline_config["performance"]["r2"],
        mae=baseline_config["performance"]["mae"]
    )
    
    state = MMMState(
        iteration=0,
        performance=performance,
        model_config=baseline_config["model_config"],
        business_context=business_context,
        convergence_score=0.0
    )
    
    return state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MMM multi-agent optimization session"
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        required=True,
        help="Path to baseline model configuration JSON"
    )
    parser.add_argument(
        "--business-context",
        type=str,
        required=True,
        help="Path to business context JSON"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum optimization iterations (default: 20)"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.95,
        help="Convergence score threshold [0, 1] (default: 0.95)"
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=None,
        help="Optional path to trained BERT router checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save optimization results (default: ./results)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    baseline_config = load_baseline_config(args.baseline_config)
    business_context = load_business_context(args.business_context)
    
    # Create baseline state
    baseline_state = create_baseline_state(baseline_config, business_context)
    
    # Initialize orchestrator
    print(f"Initializing orchestrator...")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Convergence threshold: {args.convergence_threshold}")
    print(f"  BERT model: {args.bert_model or 'Rule-based fallback'}")
    print()
    
    orchestrator = Orchestrator(
        baseline_state=baseline_state,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        bert_model_path=args.bert_model
    )
    
    # Run optimization session
    results = orchestrator.run_session()
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Status: {results['session_status']}")
    print(f"Iterations completed: {results['iterations_completed']}")
    print(f"Final MAPE: {results['final_state'].performance.mape:.2f}%")
    print(f"Final R²: {results['final_state'].performance.r2:.3f}")
    print(f"Convergence score: {results['convergence_score']:.2f}")
    print(f"Stakeholder decision: {results['stakeholder_decision']}")
    print(f"Termination reason: {results['termination_reason']}")
    print()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "optimization_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "status": results["session_status"],
            "iterations": results["iterations_completed"],
            "final_mape": results["final_state"].performance.mape,
            "final_r2": results["final_state"].performance.r2,
            "convergence_score": results["convergence_score"],
            "stakeholder_decision": results["stakeholder_decision"],
            "termination_reason": results["termination_reason"]
        }, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    
    # Save full history
    history_file = output_dir / "optimization_history.json"
    history_json = results["history"].to_json()
    with open(history_file, "w") as f:
        f.write(history_json)
    
    print(f"✓ Full history saved to: {history_file}")


if __name__ == "__main__":
    main()
