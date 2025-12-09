# MMM Multi-Agent Optimization System

A sophisticated multi-agent system for optimizing Marketing Mix Models (MMM) using intelligent agent coordination, BERT-based routing, and methodology protection.

## 🎯 Overview

This system orchestrates four specialized AI agents to collaboratively optimize MMM hyperparameters while protecting core methodology integrity. It implements a zero-tolerance approach to methodology changes through an automated Research Persona guardian.

### Key Features

- **🤖 Multi-Agent Architecture**: 4 specialized agents (Data Scientist, Business Analyst, Stakeholder, Research Persona)
- **🧠 BERT-Based Routing**: Intelligent agent selection using fine-tuned BERT embeddings
- **🛡️ Methodology Protection**: Zero-tolerance system for protecting MMM best practices
- **🔄 Rejection Handling**: Hybrid escalation strategy (1st: route back, 2nd: alternative, 3rd+: human)
- **📊 Complete Audit Trail**: Full history tracking with MMMHistory
- **🔌 A2A Protocol**: Agent-to-Agent communication via JSON-RPC (extensible to HTTP)

## 📁 Project Structure

```
mmm_optimizer/
├── __init__.py                      # Package initialization
├── orchestrator/
│   ├── __init__.py
│   ├── state.py                     # MMMState, IterationRecord, MMMHistory
│   └── orchestrator.py              # Main orchestration engine
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                # BaseAgent abstract class
│   ├── data_scientist_agent.py      # Hyperparameter tuning
│   ├── business_analyst_agent.py    # Business validation
│   ├── stakeholder_agent.py         # Executive approval
│   └── research_persona_agent.py    # Methodology guardian (always rejects)
├── methodology_guard/
│   ├── __init__.py
│   ├── methodology_spec.json        # Protected vs tunable fields
│   ├── detector.py                  # Layer 1: Config comparison
│   └── rejection_handler.py         # Hybrid rejection strategy
├── routing/
│   ├── __init__.py
│   └── bert_router.py               # BERT-based agent selection
├── a2a/
│   ├── __init__.py
│   ├── registry.json                # Agent endpoint registry
│   └── client.py                    # A2A communication client
├── entity_intent/
│   ├── __init__.py
│   └── classifier.py                # Entity/intent extraction (optional)
├── scripts/
│   ├── __init__.py
│   ├── run_session.py               # Main entry point
│   └── train_bert_router.py         # BERT training script
├── config/
│   ├── __init__.py
│   └── settings.yaml                # System configuration
└── examples/
    ├── baseline_model.json          # Example baseline config
    └── business_context.json        # Example business context
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd My-awsome-project/mmm_optimizer

# Install dependencies (requirements.txt to be created)
pip install -r requirements.txt
```

### 2. Run Optimization Session

```bash
python scripts/run_session.py \
    --baseline-config examples/baseline_model.json \
    --business-context examples/business_context.json \
    --max-iterations 20 \
    --convergence-threshold 0.95
```

### 3. Output

Results will be saved to `./results/`:
- `optimization_results.json` - Summary metrics
- `optimization_history.json` - Complete audit trail

## 🏗️ Architecture

### 5-Phase Workflow

1. **Technical Optimization** (Data Scientist)
   - Analyzes MAPE, R², MAE
   - Tunes hyperparameters within allowed ranges
   - Proposes parameter updates

2. **Methodology Validation** (Research Persona)
   - Auto-triggered after every DS proposal
   - Checks for methodology violations
   - **ALWAYS REJECTS** methodology changes
   - Provides academic justification

3. **Business Validation** (Business Analyst)
   - Validates ROI against industry benchmarks
   - Checks carryover duration realism
   - Flags warnings/critical issues

4. **Rejection Handling**
   - 1st rejection → Route back to originating agent with feedback
   - 2nd rejection → Route to alternative agent
   - 3rd+ rejection → Human escalation

5. **Executive Approval** (Stakeholder)
   - Final deployment decision
   - Reviews optimization summary
   - Sets success criteria and monitoring

### Agent Responsibilities

| Agent | Role | Can Change | Cannot Change |
|-------|------|------------|---------------|
| Data Scientist | Hyperparameter tuning | Decay rates, alpha values, learning rates | Adstock type, saturation type, architecture |
| Business Analyst | Business validation | Flag issues, propose constraints | Model config, hyperparameters |
| Stakeholder | Deployment approval | Approve/reject deployment | Model config, hyperparameters |
| Research Persona | Methodology guardian | **NOTHING** (always rejects) | **EVERYTHING** (protects all methodology) |

## 🧪 Current Status

### ✅ Implemented (Stub/Framework)

- Complete project structure (all directories and files)
- State management (MMMState, IterationRecord, MMMHistory)
- All 4 specialized agents (detailed stubs)
- Methodology specification and detector
- Rejection handler with hybrid strategy
- A2A client (local calls)
- BERT router (rule-based fallback)
- Main orchestrator
- Entry scripts (run_session.py, train_bert_router.py)
- Configuration files

### 🚧 To Be Implemented

1. **LLM Integration**
   - Connect agents to Claude API (Anthropic)
   - Implement prompt engineering for each agent
   - Add retry logic and error handling

2. **BERT Training**
   - Collect historical routing data
   - Fine-tune BERT on classification task
   - Evaluate and deploy trained model

3. **Model Integration**
   - Connect to actual MMM training pipeline
   - Implement `_apply_agent_changes()` logic
   - Re-train model after hyperparameter updates

4. **Layer 2 Methodology Guard**
   - LLM-based semantic violation detection
   - Parse agent reasoning for methodology intent

5. **HTTP A2A Protocol**
   - Deploy agents as HTTP services
   - Implement JSON-RPC client
   - Add authentication and retry logic

## 📊 Example Usage

### Python API

```python
from mmm_optimizer.orchestrator.state import MMMState, Performance
from mmm_optimizer.orchestrator.orchestrator import Orchestrator

# Create baseline state
baseline = MMMState(
    iteration=0,
    performance=Performance(mape=15.2, r2=0.82, mae=1250.5),
    model_config={...},
    business_context={...},
    convergence_score=0.0
)

# Run optimization
orchestrator = Orchestrator(
    baseline_state=baseline,
    max_iterations=20,
    convergence_threshold=0.95
)

results = orchestrator.run_session()

print(f"Final MAPE: {results['final_state'].performance.mape:.2f}%")
print(f"Decision: {results['stakeholder_decision']}")
```

### CLI

```bash
# Standard run
python scripts/run_session.py \
    --baseline-config examples/baseline_model.json \
    --business-context examples/business_context.json

# With custom thresholds
python scripts/run_session.py \
    --baseline-config examples/baseline_model.json \
    --business-context examples/business_context.json \
    --max-iterations 30 \
    --convergence-threshold 0.90

# With trained BERT model
python scripts/run_session.py \
    --baseline-config examples/baseline_model.json \
    --business-context examples/business_context.json \
    --bert-model ./checkpoints/bert_router_v1
```

## 🎓 Design Document

Full technical specification: See `optimization.md` (provided by user)

Key sections:
- Section 3: Multi-agent architecture
- Section 4: Research Persona (methodology guardian)
- Section 5: Rejection handling strategy
- Section 8: BERT routing architecture
- Section 9: A2A protocol specification

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

- Max iterations and convergence thresholds
- Agent timeouts and rejection limits
- BERT routing vs rule-based fallback
- Business validation benchmarks
- Logging and audit settings

## 📝 Requirements

To be created (`requirements.txt`):

```
anthropic>=0.x.x          # Claude API
transformers>=4.30.0      # BERT
torch>=2.0.0              # PyTorch
pyyaml>=6.0               # Config loading
```

## 🤝 Contributing

This is a stub implementation designed for future LLM integration. Key areas for contribution:

1. Implement Claude API calls in agent `run()` methods
2. Create MMM training pipeline integration
3. Collect and label BERT training data
4. Implement Layer 2 methodology guard
5. Add comprehensive unit tests

## 📄 License

[To be determined]

## 👥 Authors

Built based on comprehensive design specification in `optimization.md`.

---

**Status**: 🚧 Stub implementation - Framework complete, LLM integration pending
