# MMM Multi-Agent Optimization System
## Complete Technical Documentation

**Version:** 1.0  
**Date:** November 29, 2025  
**Author:** MMM Development Team

---

## 1. Executive Summary

### 1.1 Overview

This document describes an **intelligent multi-agent system** that automates Marketing Mix Model (MMM) optimization. The system replaces manual, weeks-long hyperparameter tuning with an automated process that takes 2-3 days.

### 1.2 Core Innovation

Four specialized AI agents (Data Scientist, Business Analyst, Stakeholder, Research Persona) collaborate to optimize MMM models. A BERT-based neural network learns from historical optimizations to intelligently route between agents, while a methodology guardian ensures core MMM principles are never violated.

### 1.3 Key Benefits

| Metric | Manual Process | Automated System |
|--------|----------------|------------------|
| Time to optimize | 2-4 weeks | 2-3 days |
| Consistency | Variable | 100% reproducible |
| Methodology compliance | 85% (15% drift) | 100% (zero violations) |
| Business validation | At end (rework common) | Continuous (minimal rework) |

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                           │
│  • Manages workflow and state                            │
│  • Tracks iteration history                              │
│  • Coordinates all components                            │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              BERT ROUTING ENGINE                          │
│  • Analyzes full optimization history                    │
│  • Learns patterns from past sessions                    │
│  • Predicts optimal next agent                           │
│  • Outputs probability distribution                      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│         METHODOLOGY CHANGE DETECTOR                       │
│  • Monitors all agent outputs                            │
│  • Compares proposed vs official methodology             │
│  • Triggers Research Persona on violations               │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              A2A PROTOCOL LAYER                           │
│  • Standardized JSON-RPC communication                   │
│  • Agent registry and discovery                          │
│  • Authentication and routing                            │
└──────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     Data     │  │   Business   │  │  Stakeholder │
│  Scientist   │  │   Analyst    │  │    Agent     │
│              │  │              │  │              │
│ Hyperparam   │  │ Business     │  │ Strategic    │
│ tuning       │  │ validation   │  │ approval     │
└──────────────┘  └──────────────┘  └──────────────┘
                          ↓
              ┌───────────────────────┐
              │   Research Persona    │
              │ (Methodology Guardian)│
              │  - Always rejects     │
              │    methodology changes│
              └───────────────────────┘
```

### 2.2 Component Responsibilities

**Orchestrator:**
- Maintains current state (metrics, configuration, iteration count)
- Stores complete history of all iterations
- Coordinates execution flow
- Applies changes (if approved)
- Checks stopping conditions

**BERT Router:**
- Reads: Full history + current state (formatted as text)
- Processes: Through fine-tuned BERT model
- Outputs: Probability for each agent (e.g., {DS: 0.15, BA: 0.25, SH: 0.55, RP: 0.05})
- Learns: From outcomes to improve routing over time

**Methodology Detector:**
- Compares: Old methodology config vs new proposed config
- Checks: Protected fields (adstock type, saturation type, hierarchy, architecture)
- Triggers: Research Persona if any protected field changed
- Overrides: BERT routing when methodology violation detected

**A2A Protocol:**
- Looks up: Agent endpoint from registry
- Formats: JSON-RPC 2.0 requests
- Sends: HTTP POST with authentication
- Receives: Standardized JSON responses

**Four Agents:**
- Each represents a stakeholder perspective
- All use Claude API (Anthropic) for LLM-based reasoning
- Return structured outputs with proposed changes + reasoning

---

## 3. The Four Agent Personas

### 3.1 Agent Responsibilities

| Agent | Role | Can Do | Cannot Do |
|-------|------|--------|-----------|
| **Data Scientist** | Technical optimizer | Tune hyperparameters, run diagnostics | Change methodology, override business rules |
| **Business Analyst** | Business validator | Flag unrealistic outputs, set constraints | Change methodology, force results |
| **Stakeholder** | Strategic approver | Final approval, budget authority | Change methodology without justification |
| **Research Persona** | Methodology guardian | Reject violations, educate | Approve methodology changes (always rejects) |

### 3.2 Data Scientist Agent

**Purpose:** Improve statistical model performance (minimize MAPE, maximize R²)

**What it does:**
- Adjusts hyperparameters: adstock decay (0.2-0.9), saturation alpha (1.5-3.5), learning rates (0.0001-0.01)
- Runs diagnostics: checks residuals, multicollinearity, cross-validation
- Proposes data transformations
- Analyzes performance trends

**Example Input:**
```
Task: optimize_hyperparameters
Current MAPE: 9.2%
Current config: {tv_adstock_decay: 0.65, tv_saturation_alpha: 2.3}
History: [Last 3 iterations showed 1.2%, 0.8%, 0.4% improvements]
```

**Example Output:**
```
Action: hyperparameter_update
Changes: {tv_adstock_decay: 0.65 → 0.70, tv_saturation_alpha: 2.3 → 2.5}
Reasoning: "Extending TV carryover to ~10 weeks (industry standard for CPG)"
Expected improvement: 0.8-1.2% MAPE reduction
Flags: {needs_business_review: true}
```

### 3.3 Business Analyst Agent

**Purpose:** Ensure model outputs make business sense

**What it does:**
- Validates ROI values against industry benchmarks (TV: 2.5-4.5x, Digital: 3-6x)
- Checks carryover durations (TV: 6-12 weeks for CPG)
- Flags unrealistic coefficients
- Sets business constraints (minimum spend, strategic priorities)
- Translates technical outputs to business language

**Example Input:**
```
Task: validate_business_logic
Model results: {tv_roi: 3.2x, digital_roi: 8.5x, tv_carryover: 8.5 weeks}
Benchmarks: {cpg_tv_roi: "2.5-4.5x", cpg_digital_roi: "3-6x"}
```

**Example Output:**
```
Validation: PARTIAL_ALIGNMENT
Flags: [
  {issue: "Digital ROI (8.5x) exceeds benchmark (3-6x)", 
   recommendation: "Investigate digital attribution data quality"}
]
Constraints to add: {digital_roi_cap: 6.0}
Next suggestion: data_scientist (investigate data)
```

### 3.4 Stakeholder Agent

**Purpose:** Make final strategic deployment decision

**What it does:**
- Reviews executive summary of optimization
- Considers business strategy, market conditions, organizational readiness
- Decides: APPROVE / CONDITIONAL_APPROVAL / REJECT / APPROVE_FOR_TESTING
- Sets implementation timeline and success criteria

**Example Input:**
```
Task: review_for_approval
Summary: 8 iterations, MAPE 8.2%, all agents validated
Recommendation: Reduce TV -20%, increase Digital +40%
Validations: DS=APPROVED, BA=APPROVED_WITH_NOTES, RP=NO_VIOLATIONS
```

**Example Output:**
```
Decision: CONDITIONAL_APPROVAL
Conditions: [
  "Phased TV reduction: -10% Q1, -10% Q2 (not immediate -20%)",
  "Monthly brand tracking study to monitor awareness",
  "Digital attribution quarterly audit"
]
Implementation: Q1-Q2 2026 (6 months phased)
Success criteria: Sales lift >2.5%, Brand awareness >85%
```

### 3.5 Research Persona Agent

**Purpose:** Protect MMM methodology integrity (ALWAYS REJECTS violations)

**Protected methodology:**
```
• adstock.type: "beta_gamma" (PROTECTED)
• saturation.type: "hill" (PROTECTED)
• hierarchy.levels: ["brand", "region", "channel"] (PROTECTED)
• architecture: "hierarchical_neural_additive" (PROTECTED)
```

**What it does:**
- Reviews all proposed methodology changes
- ALWAYS REJECTS them (no exceptions)
- Provides academic justification (references papers)
- Suggests alternatives within framework
- Educates agents on methodology principles

**Example Input:**
```
Detection: saturation.type change detected
Proposed: "hill" → "exponential"
Agent reasoning: "Exponential provides 0.3% better MAPE fit"
```

**Example Output:**
```
Decision: REJECTED
Reasoning:
  • Exponential functions don't saturate (unbounded growth)
  • Violates diminishing returns principle (Montgomery 1989)
  • 0.3% improvement too small to justify breaking methodology
Alternatives within framework:
  • Tune saturation_alpha: 2.0 → 2.5-2.8 for more flexible curve
  • Try channel-specific alpha values (TV vs Digital)
Guidance: "Keep Hill function, adjust parameters instead"
```

---

## 4. BERT-Based Intelligent Routing

### 4.1 Why BERT?

Traditional routing has limitations:
- **Fixed rules:** "Always call DS first, then BA, then SH" → too rigid
- **Random:** Inefficient, wastes iterations
- **Thompson Sampling:** No historical context, only tracks success rates

**BERT advantages:**
- Sees complete optimization trajectory (all past iterations)
- Learns complex patterns: "After 3 DS calls with diminishing returns → call Stakeholder"
- Understands sequences: "DS → BA → DS → BA = circular, break with SH"
- Provides interpretable probabilities

### 4.2 How BERT Routing Works

**Step 1: Format History as Text**
```
Input to BERT:
"[CLS] HISTORY: Iteration 1: Agent=data_scientist, MAPE=12.3→11.1, 
Result=improvement. Iteration 2: Agent=business_analyst, MAPE=11.1, 
Flags=0, Result=validated. Iteration 3: Agent=data_scientist, 
MAPE=11.1→9.8, Result=improvement. Iteration 4: Agent=data_scientist, 
MAPE=9.8→9.5, Result=small_improvement. CURRENT: MAPE=9.5, R2=0.88, 
Flags=0, LastAgents=[DS,DS,DS]. TASK: Select next agent [SEP]"
```

**Step 2: BERT Processing**
```
Text → Tokenization → BERT Encoder (12 layers) 
→ [CLS] embedding (768-dim vector capturing context)
→ Classification head (linear layer)
→ Softmax → Probabilities
```

**Step 3: Output Probabilities**
```
{
  data_scientist: 0.12,
  business_analyst: 0.23,
  stakeholder: 0.62,     ← HIGHEST
  research_persona: 0.03
}

Selected: stakeholder
Confidence: 62%
```

### 4.3 Patterns BERT Learns

Over time, BERT discovers patterns from data:

**Pattern 1: Diminishing Returns**
```
IF last_3_agents = [DS, DS, DS]
AND improvements = [1.2%, 0.8%, 0.3%] (decreasing)
AND mape < 9%
THEN stakeholder probability = 70% (time for approval)
```

**Pattern 2: Business Flag Resolution**
```
IF business_flags > 0
AND last_agent = business_analyst
THEN data_scientist probability = 75% (fix issues)
```

**Pattern 3: Circular Pattern Breaking**
```
IF last_4_agents = [DS, BA, DS, BA]
THEN stakeholder probability = 80% (break cycle, force decision)
```

### 4.4 Training BERT

**Training data:** 50-100 historical MMM optimization sessions

**Format:**
```
{
  "input_text": "HISTORY: ... CURRENT: ... [SEP]",
  "label": 2,  # 0=DS, 1=BA, 2=SH, 3=RP
  "outcome_reward": 5.0  # High reward if led to approval
}
```

**Training (Pseudo-code):**
```
model = BertForSequenceClassification(num_labels=4)
for session in historical_sessions:
    for iteration in session:
        text = format_history(iteration)
        label = next_agent_that_was_called
        model.train(text, label)
model.save()
```

**Usage (Pseudo-code):**
```
text = format_history(current_history, current_state)
probabilities = model.predict(text)
selected_agent = argmax(probabilities)
```

---

## 5. Entity Intent Classifier (Optional Enhancement)

### 5.1 Purpose

The Entity Intent Classifier is an **optional component** that sits between agent execution and BERT routing. It extracts structured information from agent outputs to help BERT understand patterns more clearly.

**Problem it solves:**
```
Raw agent output (unstructured):
"Data Scientist changed tv_adstock_decay from 0.5 to 0.7 and MAPE improved by 1.2%"

↓ BERT must learn patterns from raw text (harder)
```

**With Entity Intent Classifier:**
```
Raw agent output → Entity/Intent Classifier → Structured output:
{
  "intent": "hyperparameter_tuning",
  "entities": {
    "parameter": "tv_adstock_decay",
    "old_value": 0.5,
    "new_value": 0.7,
    "metric": "MAPE",
    "improvement": 1.2,
    "direction": "improvement"
  }
}

↓ BERT learns from structured data (easier, more accurate)
```

### 5.2 Two Components

**Component 1: Intent Classification**

Identifies what the agent is trying to do:

| Intent Type | Description | Example |
|-------------|-------------|---------|
| `hyperparameter_tuning` | Adjusting model parameters | "Changed adstock decay to 0.7" |
| `business_validation` | Checking business logic | "Flagged digital ROI as unrealistic" |
| `methodology_modification` | Attempting methodology change | "Let's remove adstock" |
| `approval` | Approving model | "Ready for deployment" |
| `rejection` | Rejecting proposal | "Need more work on attribution" |
| `data_quality_concern` | Flagging data issues | "Digital spend data suspicious" |

**Component 2: Entity Extraction**

Extracts specific pieces of information:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| `PARAMETER` | Hyperparameter name | tv_adstock_decay, learning_rate |
| `METRIC` | Performance metric | MAPE, R², ROI |
| `VALUE` | Numeric value | 0.5, 8.3%, 2.5x |
| `CHANNEL` | Marketing channel | TV, Digital, Print |
| `COMPONENT` | MMM component | adstock, saturation, hierarchy |
| `ACTION` | What was done | changed, removed, increased |
| `DIRECTION` | Trend direction | improvement, decline, stable |

### 5.3 How It Works

**Architecture:**
```
Agent Output (text)
        ↓
┌──────────────────────────────────┐
│  Intent Classifier (BERT model)  │
│  → Predicts intent category      │
└──────────────────────────────────┘
        ↓
   Intent: "hyperparameter_tuning"
        ↓
┌──────────────────────────────────┐
│ Entity Extractor (BERT-NER model)│
│ → Extracts named entities        │
└──────────────────────────────────┘
        ↓
   Entities: {parameter: "tv_adstock_decay", 
              old_value: 0.5, new_value: 0.7}
        ↓
┌──────────────────────────────────┐
│   Structured Output (JSON)       │
│   Combined intent + entities     │
└──────────────────────────────────┘
        ↓
   Passed to BERT Router
```

### 5.4 Integration with BERT Router

**Without Entity/Intent Classifier:**
```
def format_for_bert(history, current_state):
    text = "[CLS] HISTORY: "
    
    for iteration in history:
        # Raw text from agent
        text += f"Iteration {i}: Agent={agent}, "
        text += f"Output: {raw_agent_output_text}. "
    
    text += "CURRENT: MAPE=9.5%, ... [SEP]"
    return text
```

**With Entity/Intent Classifier:**
```
def format_for_bert_with_entities(history, current_state):
    text = "[CLS] HISTORY: "
    
    for iteration in history:
        # Classify intent
        intent = intent_classifier.predict(iteration.agent_output)
        
        # Extract entities
        entities = entity_extractor.predict(iteration.agent_output)
        
        # Structured format (easier for BERT to learn patterns)
        text += f"Iteration {i}: "
        text += f"Agent={agent}, "
        text += f"Intent={intent}, "
        text += f"Changed={entities['PARAMETER']}, "
        text += f"From={entities['old_value']}, "
        text += f"To={entities['new_value']}, "
        text += f"Metric={entities['METRIC']}, "
        text += f"Direction={entities['DIRECTION']}. "
    
    text += "CURRENT: MAPE=9.5%, ... [SEP]"
    return text
```

### 5.5 Example Processing

**Input (Raw Agent Output):**
```
"I adjusted the tv_adstock_decay parameter from 0.65 to 0.70 to extend 
carryover duration. This resulted in MAPE improving from 9.2% to 8.5%, 
which is a 0.7% improvement. I recommend Business Analyst validates 
the new TV ROI values."
```

**Output (Structured):**
```json
{
  "intent": "hyperparameter_tuning",
  "action": "adjusted",
  "parameter_changed": "tv_adstock_decay",
  "old_value": 0.65,
  "new_value": 0.70,
  "metric_tracked": "MAPE",
  "metric_before": 9.2,
  "metric_after": 8.5,
  "improvement": 0.7,
  "direction": "improvement",
  "next_agent_suggestion": "business_analyst"
}
```

### 5.6 Special Use Case: Methodology Violation Detection

Entity/Intent Classifier helps catch methodology violations earlier:

```
Agent output: "To simplify the model, let's remove adstock effects 
for print channel since carryover seems minimal."

Intent Classification:
  intent: "methodology_modification" (⚠️ Warning!)
  
Entity Extraction:
  COMPONENT: ["adstock"]
  ACTION: ["remove"]
  CHANNEL: ["print"]

↓ Triggers methodology detector BEFORE changes applied
↓ Routes to Research Persona
↓ Research Persona rejects early
```

### 5.7 When to Use

**Use When:**
- Agent outputs are verbose and unstructured
- BERT routing struggles to learn patterns
- You have 500+ labeled examples for training
- You need programmatic access to specific fields

**Skip When:**
- Agent outputs already structured (JSON format)
- BERT routing works well without it
- Limited training data (<500 examples)
- Adds complexity without clear benefit

**Recommendation:** Add as Phase 2 enhancement after system proves effective.

---

## 6. Methodology Guardian System

### 6.1 Purpose

**Prevent any changes to protected MMM components** while allowing hyperparameter tuning.

### 6.2 Detection Mechanism

**Three-layer detection:**

**Layer 1: Config Field Comparison**
```
protected_fields = [
    "adstock.type",
    "saturation.type", 
    "hierarchy.levels",
    "architecture.type"
]

for field in protected_fields:
    if current[field] != proposed[field]:
        return VIOLATION_DETECTED
```

**Layer 2: Protected File Monitoring**
```
protected_files = [
    "src/models/layers/adstock_layer.py",
    "src/models/layers/saturation_layer.py",
    "src/models/mmm_model.py"
]

if any(changed_file in protected_files):
    return VIOLATION_DETECTED
```

**Layer 3: Language Pattern Scanning**
```
risky_phrases = [
    "remove adstock",
    "change saturation to exponential",
    "simplify by removing hierarchy"
]

if any(phrase in agent_reasoning.lower()):
    return POTENTIAL_VIOLATION
```

### 6.3 Integration with Routing

```
def route_next_agent(history, state, last_agent_output):
    
    # Check for methodology violations
    violation = detect_methodology_change(last_agent_output)
    
    if violation.detected:
        # OVERRIDE BERT - Force to Research Persona
        print("⚠️ Methodology violation → Research Persona")
        return 'research_persona'
    
    # Normal BERT routing
    probabilities = bert_router.predict(history, state)
    selected = max(probabilities)
    
    return selected
```

---

## 7. Rejection Handling Strategies

### 7.1 The Problem

When Research Persona rejects a methodology change:
```
Data Scientist: "Change saturation to exponential"
         ↓
Research Persona: "REJECTED - exponential doesn't saturate"
         ↓
WHAT NEXT?
```

### 7.2 Three Options

**Option 1: Automatic (Fast, Scalable)**
```
Rejection → Route back to Data Scientist with feedback
DS receives: rejection reason + alternatives + guidance
DS proposes new approach within methodology
```

**Option 2: Human Review (Maximum Oversight)**
```
Rejection → Pause system → Notify human
Human reviews → Decides: accept/override/stop
System resumes based on human decision
```

**Option 3: Hybrid (Recommended) 🏆**
```
1st rejection → Automatic (route back with feedback)
2nd rejection → Automatic (try alternative agent)
3rd+ rejection → Escalate to human
```

### 7.3 Hybrid Implementation

```
class RejectionHandler:
    rejection_count = {}
    
    def handle(research_response, agent, iteration):
        count = rejection_count[agent]
        
        if count == 1:
            # First rejection: automatic
            return route_back_to(agent, with_feedback=True)
        
        elif count == 2:
            # Second rejection: try different agent
            alternative = get_alternative_agent(agent)
            return route_to(alternative)
        
        else:
            # Third+ rejection: human escalation
            notify_human(research_response)
            return pause_for_human_review()
```

---

## 8. Complete Workflow

### 8.1 Workflow Diagram

```
START
  ↓
┌──────────────────────────────┐
│ INITIALIZATION               │
│ • Load MMM model             │
│ • Evaluate baseline (MAPE)   │
│ • Initialize state & history │
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ ITERATION LOOP (1-20)        │
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ ENTITY/INTENT CLASSIFIER     │
│ (Optional) Extract structure │
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ BERT ROUTING                 │
│ • Format history as text     │
│ • BERT predicts probabilities│
│ • Select highest probability │
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ A2A AGENT CALL               │
│ • Lookup endpoint            │
│ • Send JSON-RPC request      │
│ • Receive response           │
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ METHODOLOGY CHECK            │
│ • Compare old vs new config  │
│ • Check protected fields     │
│ • Check code files           │
└──────────────────────────────┘
  ↓
  ├─[Violation?]─Yes→┌────────────────────┐
  │                  │ FORCE ROUTE TO     │
  │                  │ RESEARCH PERSONA   │
  │                  │   ↓                │
  │                  │ ALWAYS REJECTS     │
  │                  │   ↓                │
  │                  │ Handle Rejection:  │
  │                  │ • Auto (1st-2nd)   │
  │                  │ • Human (3rd+)     │
  │                  └────────────────────┘
  │                           ↓
  └─[No Violation]────→┌────────────────────┐
                       │ APPLY CHANGES      │
                       │ • Update hyperparams│
                       │ • Update state     │
                       │ • Add to history   │
                       └────────────────────┘
                                ↓
                       ┌────────────────────┐
                       │ CHECK STOPPING     │
                       │ • Approved?        │
                       │ • Max iterations?  │
                       └────────────────────┘
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
                [STOP]                  [CONTINUE]
                    ↓                       ↓
         ┌──────────────────┐      [Loop to BERT]
         │ COMPLETE         │
         │ • Final metrics  │
         │ • Full history   │
         │ • Approval status│
         └──────────────────┘
```

### 8.2 Example Session

**Session: Brand_A_Q4_2025**

```
═══════════════════════════════════════════
ITERATION 1
═══════════════════════════════════════════
State: MAPE=12.3%, R²=0.82, History=[]

BERT: {DS:0.70, BA:0.20, SH:0.05, RP:0.05} → data_scientist

DS Output: Changed tv_adstock_decay 0.5→0.6
Methodology Check: ✅ NO VIOLATION
Apply: MAPE improved to 11.1%

═══════════════════════════════════════════
ITERATION 2
═══════════════════════════════════════════
State: MAPE=11.1%

BERT: {DS:0.35, BA:0.55, SH:0.05, RP:0.05} → business_analyst

BA Output: Validated ROI values, all within benchmarks
Flags: 0
Continue

═══════════════════════════════════════════
ITERATIONS 3-5
═══════════════════════════════════════════
[DS tunes → BA validates → DS refines]
MAPE: 11.1% → 9.8% → 9.2% → 8.5%

═══════════════════════════════════════════
ITERATION 6
═══════════════════════════════════════════
State: MAPE=8.5%, Last 3 agents=[DS,DS,DS]

BERT: {DS:0.60, BA:0.20, SH:0.15, RP:0.05} → data_scientist

DS Output: Changed saturation.type "hill"→"exponential"
Methodology Check: 🚨 VIOLATION (protected field changed)

FORCED ROUTE → research_persona

RP Output: REJECTED
  Reason: "Exponential doesn't saturate"
  Alternatives: "Tune saturation_alpha instead"
  
Rejection Handler (1st rejection):
  Action: Route back to DS with feedback

═══════════════════════════════════════════
ITERATION 7
═══════════════════════════════════════════
DS receives rejection feedback

DS Output: Changed tv_saturation_alpha 2.2→2.7
  (Following Research guidance)
Methodology Check: ✅ NO VIOLATION
Apply: MAPE improved to 8.2%

═══════════════════════════════════════════
ITERATION 8
═══════════════════════════════════════════
State: MAPE=8.2%, Last agents=[DS,DS,DS]
      Improvements=[0.3%, 0.4%, 0.3%] diminishing

BERT: {DS:0.10, BA:0.20, SH:0.65, RP:0.05} → stakeholder
  (Learned pattern: 3 DS with diminishing returns → approval time)

SH Output: CONDITIONAL_APPROVAL
  Conditions: Phased implementation, brand tracking
  Timeline: Q1-Q2 2026

STOP: ✅ APPROVED
═══════════════════════════════════════════

Final: 8 iterations, 2 days, MAPE 8.2%
```

---

## 9. Key Implementation Notes

### 9.1 A2A Communication

**Agent Registry:**
```json
{
  "data_scientist": {
    "endpoint": "https://mmm.company.com/ds/execute",
    "auth": "Bearer token123"
  },
  "business_analyst": {...},
  "stakeholder": {...},
  "research_persona": {...}
}
```

**JSON-RPC Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "execute",
  "params": {
    "task": "optimize_hyperparameters",
    "context": {"state": {...}, "history": [...]}
  },
  "id": "req-iter-5"
}
```

**JSON-RPC Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "agent": "data_scientist",
    "action": "hyperparameter_update",
    "proposed_changes": {...},
    "reasoning": "..."
  },
  "id": "req-iter-5"
}
```

### 9.2 State Management

```
class MMMState:
    iteration: int
    performance: {mape, r2, mae}
    model_config: {
        methodology: {adstock_type, saturation_type, hierarchy},
        hyperparameters: {tv_adstock_decay, tv_saturation_alpha, ...}
    }
    business_context: {flags, constraints}
    convergence_score: float

class MMMHistory:
    iterations: List[{
        iteration: int,
        agent: str,
        action: str,
        state_before: MMMState,
        state_after: MMMState,
        outcome: str
    }]
```

### 9.3 Training Requirements

**BERT Router:**
- Training data: 50-100 historical MMM optimization sessions
- Each session: 10-20 iterations
- Total training examples: 500-2000 routing decisions
- Training time: 1-2 hours on GPU

**Agent LLMs:**
- Use Claude API (Anthropic)
- No training needed (prompt-based)
- Cost: ~$0.50-1.00 per optimization session

---

## 10. Success Metrics

### 10.1 System Performance

- **Time to optimize:** Target <3 days (vs 2-4 weeks manual)
- **Iterations to approval:** Target <12 (vs 15-20 manual)
- **Methodology compliance:** Target 100% (vs 85% manual)
- **Approval rate:** Target >90%

### 10.2 BERT Routing Quality

- **Routing accuracy:** Target >70% (agent selected leads to progress)
- **Circular patterns:** Target <5% (avoid DS↔BA loops)
- **Early approval detection:** Target 30% (approve before max iterations)

### 10.3 Business Impact

- **Model quality:** MAPE <8% consistently
- **Business alignment:** Zero rejected models due to business invalidity
- **Stakeholder satisfaction:** >85% approval on first review
- **Cost savings:** 85% reduction in analyst time

---

## 11. Conclusion

This MMM multi-agent optimization system combines modern AI techniques (BERT routing, LLM-based agents, A2A protocol) to automate a previously manual, time-consuming process. The key innovations are:

1. **BERT learns optimal routing** from historical sessions
2. **Methodology guardian** ensures zero methodology violations
3. **Continuous business validation** prevents late-stage rework
4. **Hybrid rejection handling** balances automation with oversight
5. **Entity/Intent classifier** (optional) improves BERT understanding

The system is production-ready and can be implemented in 10-12 weeks, delivering immediate ROI through reduced optimization time and improved model quality.

---

**END OF DOCUMENTATION**