# Dynamic BERT Router - Online Learning Guide

## Overview

The **Dynamic BERT Router** implements online learning (continual learning) for intelligent agent routing. It starts with a pre-trained model from synthetic data and continuously improves by learning from real optimization sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNING LIFECYCLE                        │
└─────────────────────────────────────────────────────────────┘

Phase 1: COLD START (Iterations 1-5)
┌──────────────────────┐
│  Pre-trained Model   │  ← Trained on 3000+ synthetic examples
│  (Synthetic Data)    │     from train_bert_router.py
└──────────────────────┘
          │
          ▼
    Route agents with high confidence
    (Model knows patterns from simulation)


Phase 2: DATA COLLECTION (All iterations)
┌──────────────────────┐
│  Real Optimization   │
│      Session         │
└──────────────────────┘
          │
          ▼
    Every iteration becomes a training example:
    - State text (BERT input)
    - Agent selected
    - Actual outcome
    - Stored in training_data.json


Phase 3: DYNAMIC RETRAINING (Every N iterations)
┌──────────────────────┐
│  Fine-tune Model     │  ← Real data + Synthetic data
│  (Real + Synthetic)  │     Conservative learning rate
└──────────────────────┘
          │
          ▼
    Model adapts to:
    - Your specific industry patterns
    - Your business constraints
    - Your optimization style


Phase 4: CONTINUOUS IMPROVEMENT
┌──────────────────────┐
│  Improved Router     │  ← Better at routing for YOUR use case
│  (Domain-Adapted)    │     Learns from mistakes
└──────────────────────┘
```

## Key Features

### 1. **Cold Start with Synthetic Model**
- Starts with pre-trained BERT from `bert_router_trained/`
- Model trained on 3000+ synthetic optimization sessions
- Understands MMM patterns: MAPE trends, BA-DS loops, convergence signals
- **Advantage**: Immediate intelligent routing from iteration 1

### 2. **Online Learning**
- Records every iteration as training example
- Stores: state, agent used, outcome, metadata
- Persists to `data/bert_router_training.json`
- **Advantage**: Learns from your real optimization patterns

### 3. **Automatic Retraining**
- Triggers every N iterations (default: 10)
- Fine-tunes model with accumulated real data
- Uses conservative learning rate (1e-5) to prevent forgetting
- **Advantage**: Model continuously improves

### 4. **Catastrophic Forgetting Prevention**
- Blends synthetic + real data during retraining
- Maintains baseline knowledge while adapting
- Conservative hyperparameters (2 epochs, low LR)
- **Advantage**: Doesn't lose general MMM knowledge

## Usage

### Basic Usage

```python
from mmm_optimizer.routing.dynamic_bert_router import DynamicBERTRouter

# Initialize with pre-trained model
router = DynamicBERTRouter(
    pretrained_model_path="./mmm_optimizer/bert_router_trained",
    retrain_interval=10,  # Retrain every 10 iterations
    min_examples_for_retrain=5
)

# Each iteration
next_agent = router.route(state, history)

# After agent execution, record for learning
router.record_iteration(
    state=state,
    history=history,
    actual_agent_used="data_scientist",
    outcome={"mape": 12.5, "r2": 0.88}
)
# Automatically triggers retraining when threshold reached
```

### With Orchestrator

```python
from mmm_optimizer.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    baseline_state=initial_state,
    use_dynamic_learning=True,  # Enable online learning
    retrain_interval=10,         # Retrain every 10 iterations
    max_iterations=20
)

results = orchestrator.run_session()

# Get learning statistics
stats = orchestrator.dynamic_router.get_statistics()
print(f"Collected {stats['total_examples']} examples")
print(f"Retrained {stats['retrain_count']} times")
```

### Export Training Data

```python
# Export collected data for analysis
router.export_training_data("./my_training_data.json")
```

## Configuration

### Retraining Triggers

```python
router = DynamicBERTRouter(
    retrain_interval=10,           # Retrain every N iterations
    min_examples_for_retrain=5     # Minimum examples needed
)
```

### Training Hyperparameters

```python
router.retrain(
    epochs=2,                # Conservative: avoid overfitting
    learning_rate=1e-5,      # Very low: preserve base knowledge
    batch_size=8             # Small batches for stable gradients
)
```

### Data Management

```python
router = DynamicBERTRouter(
    max_training_examples=500,     # FIFO: keep recent 500 examples
    use_synthetic_data=True        # Blend synthetic during retraining
)
```

## Training Data Format

Each training example contains:

```json
{
  "iteration": 5,
  "session_id": "20241208_143022",
  "timestamp": "2024-12-08T14:32:15",
  "state_text": "Iteration 5: MAPE=12.3%, R2=0.87, MAE=950. Last 3 agents: data_scientist, business_analyst, data_scientist. Business flags: 1. Hyperparameters: tv_adstock_decay=0.65, digital_saturation_alpha=2.4. Industry: CPG.",
  "agent_label": 0,
  "actual_outcome": {
    "mape": 11.8,
    "r2": 0.88,
    "decision": "hyperparameter_update"
  },
  "metadata": {
    "mape": 12.3,
    "r2": 0.87,
    "convergence": 0.45
  }
}
```

## Advantages Over Static Router

| Feature | Static Router | Dynamic Router |
|---------|--------------|----------------|
| **Initial Performance** | Good (synthetic data) | Good (synthetic data) |
| **Adaptation** | None | Learns from your data |
| **Industry-Specific** | General patterns | Adapts to your industry |
| **Mistake Correction** | Repeats mistakes | Learns from failures |
| **Data Efficiency** | Requires large dataset | Improves with few examples |
| **Deployment** | Train once, use forever | Continuously improves |

## Example Scenario

**Iteration 1-5**: Uses pre-trained model
- Router knows general MMM patterns
- Routes based on synthetic training
- Confidence: 75-85%

**Iteration 6**: Collects 5 real examples
- Notices your BA is stricter than training data
- Records BA rejections with context

**Iteration 10**: First retraining
- Fine-tunes on 10 real examples
- Learns your BA prefers DS to optimize more before validation
- Model adjusts routing probabilities

**Iteration 15**: Second retraining
- Now has 15 real examples
- Recognizes your industry-specific convergence patterns
- Confidence improves: 85-95%

**Iteration 20+**: Fully adapted
- Model optimized for your specific use case
- Routes like an expert who knows your business
- Minimal unnecessary BA-DS loops

## Monitoring

### Check Learning Progress

```python
stats = router.get_statistics()
print(f"Training examples: {stats['total_examples']}")
print(f"Retrain count: {stats['retrain_count']}")
print(f"Label distribution: {stats['label_distribution']}")
print(f"Next retrain in: {stats['next_retrain_in']} iterations")
```

### View Training History

```python
# Loads from data/bert_router_training.json
for example in router.training_examples[-5:]:
    print(f"Iter {example.iteration}: {router.agent_labels[example.agent_label]}")
    print(f"  MAPE: {example.metadata['mape']:.2f}%")
    print(f"  Outcome: {example.actual_outcome}")
```

## Best Practices

1. **Start with Synthetic Model**: Always initialize with pre-trained model
2. **Conservative Retraining**: Use low learning rates (1e-5) and few epochs (2)
3. **Monitor Performance**: Track routing confidence and accuracy
4. **Export Data**: Periodically save training data for analysis
5. **Blend Synthetic**: Keep `use_synthetic_data=True` to prevent forgetting
6. **Adjust Interval**: If optimization sessions are long, reduce retrain_interval

## Troubleshooting

### Router Makes Poor Decisions Early
- **Expected**: Initial model trained on synthetic data only
- **Solution**: After 5-10 iterations, model will adapt to your patterns

### Model Forgets Baseline Knowledge
- **Cause**: Too aggressive retraining (high LR, many epochs)
- **Solution**: Use `learning_rate=1e-5`, `epochs=2`, `use_synthetic_data=True`

### Training Data Not Persisting
- **Cause**: File path not writable
- **Solution**: Check `training_data_path` is writable, create parent dirs

### Retraining Takes Too Long
- **Cause**: Too many training examples
- **Solution**: Reduce `max_training_examples` to 200-300

## Technical Details

### Model Architecture
- **Base**: `bert-base-uncased` (110M parameters)
- **Fine-tuned**: 3-class classifier (DS, BA, SH)
- **Input**: Natural language state description (max 128 tokens)
- **Output**: Agent probabilities with softmax

### Learning Strategy
- **Initial**: Transfer learning from synthetic data
- **Online**: Fine-tuning with real data
- **Hybrid**: Blend synthetic + real to prevent forgetting

### Data Storage
- **Format**: JSON lines
- **Location**: `data/bert_router_training.json`
- **Size**: ~1KB per example (~500KB for 500 examples)

## Future Enhancements

- [ ] Active learning: prioritize uncertain examples
- [ ] Multi-task learning: predict outcomes too
- [ ] Confidence-based sampling: retrain on low-confidence cases
- [ ] A/B testing: compare static vs dynamic router
- [ ] Explainability: attention weights for routing decisions

## References

- **Continual Learning**: Learning without forgetting
- **Transfer Learning**: Start with general knowledge, adapt to specific domain
- **Online Learning**: Update model as new data arrives
- **Catastrophic Forgetting**: Loss of previous knowledge when learning new tasks
