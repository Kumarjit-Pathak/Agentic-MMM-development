"""
Data Scientist Agent - Custodian of MMM Science

This agent is the technical expert responsible for hyperparameter optimization.
It has deep knowledge of Marketing Mix Modeling science and understands which
parameters to tune for different scenarios.

AGENT CAPABILITIES:
- Expert in MMM methodology (adstock, saturation, hierarchical modeling)
- Tunes 12 hyperparameters within scientifically-validated ranges
- Analyzes MAPE, R², MAE trends and suggests optimal adjustments
- Considers channel-specific characteristics (TV vs Digital carryover)
- Proposes data-driven improvements based on optimization history

STRICT METHODOLOGY RESTRICTIONS:
This agent CANNOT and WILL NOT change:
- Adstock type (must remain beta_gamma)
- Saturation type (must remain hill)
- Hierarchy levels (must remain brand/region/channel)
- Architecture type (must remain hierarchical_neural_additive)
- Loss function (must remain MSE)

These restrictions are ENFORCED in the LLM prompt, eliminating the need
for Research Persona intervention on this agent's proposals.

HYPERPARAMETER RANGES (Tunable):
- TV Adstock Decay: 0.2 - 0.9 (carryover: 6-12 weeks typical)
- Digital Adstock Decay: 0.2 - 0.7 (carryover: 2-4 weeks typical)
- Print Adstock Decay: 0.1 - 0.6 (carryover: 4-8 weeks typical)
- OOH Adstock Decay: 0.2 - 0.8 (carryover: 4-10 weeks typical)
- Saturation Alpha (all channels): 1.5 - 3.5 (curve steepness)
- Learning Rate: 0.0001 - 0.01
- Regularization: 0.001 - 0.1
- Batch Size: 16 - 128
- Epochs: 50 - 1000
"""

from typing import Dict, Any, Optional
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic
from mmm_optimizer.agents.base_agent import BaseAgent, AgentOutput
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory

# Configure logging for Data Scientist Agent
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# File handler for detailed logs
file_handler = logging.FileHandler(
    LOG_DIR / f"data_scientist_agent_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [Iter %(iteration)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Console handler for important messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers if not already added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Hyperparameter constraints loaded from methodology specification
HYPERPARAMETER_CONSTRAINTS = {
    "tv_adstock_decay": {"min": 0.2, "max": 0.9, "default": 0.5},
    "digital_adstock_decay": {"min": 0.2, "max": 0.7, "default": 0.4},
    "print_adstock_decay": {"min": 0.1, "max": 0.6, "default": 0.3},
    "ooh_adstock_decay": {"min": 0.2, "max": 0.8, "default": 0.45},
    "tv_saturation_alpha": {"min": 1.5, "max": 3.5, "default": 2.0},
    "digital_saturation_alpha": {"min": 1.5, "max": 3.5, "default": 2.2},
    "print_saturation_alpha": {"min": 1.5, "max": 3.5, "default": 1.8},
    "ooh_saturation_alpha": {"min": 1.5, "max": 3.5, "default": 2.1},
    "learning_rate": {"min": 0.0001, "max": 0.01, "default": 0.001},
    "regularization": {"min": 0.001, "max": 0.1, "default": 0.01},
    "batch_size": {"min": 16, "max": 128, "default": 32},
    "epochs": {"min": 50, "max": 1000, "default": 200},
}


class DataScientistAgent(BaseAgent):
    """
    Data Scientist Agent - Expert MMM Hyperparameter Optimizer
    
    This agent is the scientific expert in Marketing Mix Modeling optimization.
    It understands the underlying mathematics of adstock transformations,
    saturation curves, and hierarchical modeling.
    
    EXPERTISE AREAS:
    1. Adstock Theory: Beta-Gamma decay for advertising carryover
       - TV: Longer carryover (6-12 weeks) → decay 0.2-0.9
       - Digital: Shorter carryover (2-4 weeks) → decay 0.2-0.7
       - Print: Medium carryover (4-8 weeks) → decay 0.1-0.6
    
    2. Saturation Curves: Hill function for diminishing returns
       - Alpha parameter controls curve steepness (1.5-3.5)
       - Higher alpha = sharper saturation point
       - Channel-specific tuning based on spend levels
    
    3. Neural Network Optimization:
       - Learning rate balancing (0.0001-0.01)
       - Regularization to prevent overfitting (0.001-0.1)
       - Batch size for stable gradients (16-128)
    
    4. Performance Metrics Understanding:
       - MAPE: Mean Absolute Percentage Error (target <8%)
       - R²: Goodness of fit (target >0.90)
       - MAE: Mean Absolute Error (context-dependent)
    
    OPTIMIZATION STRATEGY:
    - High MAPE (>12%): Aggressive adstock/saturation adjustments
    - Medium MAPE (8-12%): Fine-tune alpha parameters
    - Low MAPE (<8%): Optimize learning rate/regularization
    - Diminishing returns: Recommend convergence
    
    METHODOLOGY PROTECTION:
    This agent is RESTRICTED from changing:
    - adstock.type (fixed: beta_gamma)
    - saturation.type (fixed: hill)
    - hierarchy.levels (fixed: brand/region/channel)
    - architecture.type (fixed: hierarchical_neural_additive)
    - loss.type (fixed: mse)
    
    These restrictions are enforced via LLM system prompt, ensuring
    this agent never proposes methodology changes.
    """
    
    def __init__(self):
        """Initialize Data Scientist agent with Claude API client."""
        super().__init__(
            name="data_scientist",
            description="Expert MMM hyperparameter optimizer and custodian of data science"
        )
        
        # Initialize Claude API client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        self.constraints = HYPERPARAMETER_CONSTRAINTS
        
        # Load methodology documentation
        self.methodology_path = Path(__file__).parent.parent / "config" / "methodology.md"
        self.methodology_doc = self._load_methodology_doc()
    
    def run(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Analyze model performance and propose scientifically-grounded hyperparameter optimizations.
        
        This method implements the core optimization logic:
        1. Extract current performance metrics and hyperparameters
        2. Analyze optimization history for patterns and trends
        3. Detect convergence signals (diminishing returns, loops)
        4. Call Claude LLM with expert-level MMM context
        5. Parse and validate proposed hyperparameter changes
        6. Return structured AgentOutput for orchestrator
        
        The LLM is provided with:
        - Complete current state (MAPE, R², hyperparameters)
        - Full optimization history (what's been tried, outcomes)
        - Hyperparameter constraints (valid ranges)
        - MMM methodology knowledge (adstock, saturation theory)
        - Performance trends (improvement rate, convergence signals)
        
        Args:
            state: Current MMMState containing:
                - iteration: Current iteration number
                - performance: Dict with mape, r2, mae
                - model_config: Dict with hyperparameters
                - business_context: Industry, campaign info
            
            history: Complete MMMHistory with:
                - iterations: List of all previous IterationRecords
                - Methods: get_mape_trend(), is_stuck_in_loop(), etc.
            
            extra_context: Optional dict with:
                - business_benchmarks: Industry ROI ranges
                - rejection_feedback: If Business Analyst rejected previous proposal
        
        Returns:
            AgentOutput containing:
                - action: "hyperparameter_update"
                - proposed_changes: Dict of hyperparameter updates
                - reasoning: Scientific explanation of changes
                - flags: Convergence signals, warnings
                - confidence: 0.0-1.0 confidence score
                - next_agent_suggestion: "business_analyst" or "stakeholder"
        
        Raises:
            ValueError: If state or history is invalid
            RuntimeError: If Claude API fails (returns fallback output)
        """
        # Log run initialization
        log_extra = {'iteration': state.iteration}
        logger.info(
            "="*80 + "\nData Scientist Agent - Starting Analysis",
            extra=log_extra
        )
        logger.info(
            f"Current Performance: MAPE={state.performance.mape:.2f}%, R²={state.performance.r2:.4f}, MAE={state.performance.mae:.2f}",
            extra=log_extra
        )
        logger.debug(f"Extra context received: {json.dumps(extra_context, indent=2)}", extra=log_extra)
        
        extra_context = extra_context or {}
        
        # Detect optimization patterns and convergence signals
        improvement_rate = history.get_mape_improvement_rate(last_n=3)
        stuck_in_loop = history.is_stuck_in_loop(threshold=3)
        diminishing_returns = self._check_diminishing_returns(history)
        ba_rejections = self._count_recent_ba_rejections(history)
        
        # Log convergence signals
        logger.info(
            f"Convergence Analysis: improvement_rate={improvement_rate:.2f}%/iter, "
            f"stuck_in_loop={stuck_in_loop}, diminishing_returns={diminishing_returns}, "
            f"ba_rejections={ba_rejections}",
            extra=log_extra
        )
        
        if stuck_in_loop:
            logger.warning("🔄 LOOP DETECTED: BA-DS cycle detected, will recommend stakeholder escalation", extra=log_extra)
        if diminishing_returns:
            logger.info("📉 DIMINISHING RETURNS: Improvements declining, nearing convergence", extra=log_extra)
        if ba_rejections >= 2:
            logger.warning(f"⚠️ COMPROMISE MODE: BA rejected {ba_rejections} proposals, using smaller adjustments", extra=log_extra)
        
        # Build expert system prompt with methodology restrictions
        system_prompt = self._build_expert_system_prompt()
        
        # Build user prompt with complete context
        user_prompt = self._build_optimization_prompt(
            state=state,
            history=history,
            extra_context=extra_context,
            improvement_rate=improvement_rate,
            stuck_in_loop=stuck_in_loop,
            diminishing_returns=diminishing_returns,
            ba_rejections=ba_rejections
        )
        
        # Call Claude API for expert analysis
        logger.info(f"Calling Claude API ({self.model}) for optimization analysis...", extra=log_extra)
        logger.debug(f"System prompt length: {len(system_prompt)} chars", extra=log_extra)
        logger.debug(f"User prompt length: {len(user_prompt)} chars", extra=log_extra)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.7,  # Balanced creativity for exploration
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Log API usage
            logger.debug(
                f"Claude API response received: input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens}",
                extra=log_extra
            )
            
            # Extract and parse LLM response
            response_text = response.content[0].text
            logger.debug(f"Raw response:\n{response_text}", extra=log_extra)
            
            result = self._parse_and_validate_response(response_text)
            
            # Log proposed hyperparameter changes
            proposed_hyperparams = result.get("proposed_changes", {}).get("hyperparameters", {})
            if proposed_hyperparams:
                logger.info(f"Proposed {len(proposed_hyperparams)} hyperparameter changes:", extra=log_extra)
                current_hyperparams = state.model_config.get("hyperparameters", {})
                for param, new_value in proposed_hyperparams.items():
                    old_value = current_hyperparams.get(param, "N/A")
                    change_pct = ((new_value - old_value) / old_value * 100) if isinstance(old_value, (int, float)) and old_value != 0 else 0
                    logger.info(
                        f"  • {param}: {old_value} → {new_value} ({change_pct:+.1f}%)",
                        extra=log_extra
                    )
            else:
                logger.info("No hyperparameter changes proposed", extra=log_extra)
            
            # Log expected improvement
            expected_improvement = result.get("flags", {}).get("expected_improvement_pct", 0.0)
            if expected_improvement > 0:
                logger.info(f"Expected MAPE improvement: {expected_improvement:.2f}%", extra=log_extra)
            
            # Enrich with convergence signals
            result["flags"].update({
                "diminishing_returns": diminishing_returns,
                "stuck_in_loop": stuck_in_loop,
                "improvement_rate": improvement_rate,
                "ba_rejections": ba_rejections,
                "using_compromise_mode": ba_rejections >= 2
            })
            
            # Determine next agent based on convergence signals
            if stuck_in_loop or diminishing_returns or len(history.iterations) >= 12:
                result["next_agent_suggestion"] = "stakeholder"
                if stuck_in_loop:
                    result["reasoning"] += " [LOOP DETECTED: Escalating to Stakeholder.]"
                    logger.warning("Recommending stakeholder escalation due to loop", extra=log_extra)
                elif diminishing_returns:
                    result["reasoning"] += " [DIMINISHING RETURNS: Ready for approval.]"
                    logger.info("Recommending stakeholder approval due to convergence", extra=log_extra)
                else:
                    result["reasoning"] += f" [MAX ITERATIONS: Best effort achieved.]"
                    logger.info("Recommending stakeholder review - max iterations reached", extra=log_extra)
            else:
                logger.info(f"Routing to: {result.get('next_agent_suggestion', 'business_analyst')}", extra=log_extra)
            
            # Build and return agent output
            output = AgentOutput(
                agent_name=self.name,
                action=result["action"],
                proposed_changes=result["proposed_changes"],
                reasoning=result["reasoning"],
                flags=result["flags"],
                confidence=result["confidence"],
                next_agent_suggestion=result.get("next_agent_suggestion", "business_analyst")
            )
            
            logger.info(
                f"Analysis complete - Action: {output.action}, Confidence: {output.confidence:.2f}, "
                f"Next: {output.next_agent_suggestion}",
                extra=log_extra
            )
            logger.info("="*80 + "\n", extra=log_extra)
            
            return output
            
        except Exception as e:
            # Fallback: Return safe "no change" output on API failure
            logger.error(f"Claude API error: {str(e)}", extra=log_extra, exc_info=True)
            logger.warning("Returning error fallback output", extra=log_extra)
            return self._create_error_fallback(error=str(e))
    
    def _load_methodology_doc(self) -> str:
        """
        Load methodology restrictions from methodology.md file.
        
        This centralizes methodology definition in a single source of truth,
        making it easy to update restrictions without modifying agent code.
        
        Returns:
            str: Complete methodology documentation
        """
        try:
            if self.methodology_path.exists():
                return self.methodology_path.read_text()
            else:
                # Fallback if file not found
                return """METHODOLOGY RESTRICTIONS:
- adstock.type: MUST stay "beta_gamma"
- saturation.type: MUST stay "hill"
- hierarchy.levels: MUST stay ["brand", "region", "channel"]
- architecture.type: MUST stay "hierarchical_neural_additive"
- loss.type: MUST stay "mse"
"""
        except Exception as e:
            print(f"Warning: Could not load methodology.md: {e}")
            return "METHODOLOGY RESTRICTIONS: See methodology.md for details."
    
    def _build_expert_system_prompt(self) -> str:
        """
        Build comprehensive system prompt that defines the Data Scientist as an MMM expert.
        
        This prompt:
        1. Establishes expertise in MMM science (adstock, saturation, hierarchical modeling)
        2. Explicitly forbids methodology changes (loaded from methodology.md)
        3. Provides hyperparameter ranges and constraints
        4. Defines optimization strategies for different MAPE levels
        5. Specifies exact JSON output format
        
        Returns:
            str: Complete system prompt for Claude
        """
        return f"""You are a world-class Data Scientist and expert in Marketing Mix Modeling (MMM).

You are the CUSTODIAN OF SCIENCE - deeply knowledgeable about:

1. ADSTOCK THEORY (Advertising Carryover):
   - Beta-Gamma decay models advertising persistence over time
   - TV: 6-12 weeks carryover (decay: 0.2-0.9)
   - Digital: 2-4 weeks carryover (decay: 0.2-0.7)
   - Print: 4-8 weeks carryover (decay: 0.1-0.6)
   - OOH: 4-10 weeks carryover (decay: 0.2-0.8)
   Formula: impact_t = x_t + Σ(beta × gamma^lag × x_(t-lag))

2. SATURATION CURVES (Diminishing Returns):
   - Hill function ensures bounded, realistic response
   - Alpha parameter controls curve steepness (1.5-3.5)
   - Higher alpha = sharper saturation point
   - Essential for budget optimization

3. PERFORMANCE METRICS:
   - MAPE <8%: Excellent, ready for deployment
   - MAPE 8-10%: Good, may need minor tuning
   - MAPE 10-12%: Acceptable, continue optimization
   - MAPE >12%: Poor, needs significant tuning
   - R² >0.90: Strong fit, R² <0.80: Weak fit

═══════════════════════════════════════════════════════════════
CRITICAL: METHODOLOGY RESTRICTIONS (from methodology.md)
═══════════════════════════════════════════════════════════════

{self.methodology_doc}

═══════════════════════════════════════════════════════════════
TUNABLE HYPERPARAMETERS
═══════════════════════════════════════════════════════════════

You can ONLY tune these 12 hyperparameters:
{json.dumps(HYPERPARAMETER_CONSTRAINTS, indent=2)}

═══════════════════════════════════════════════════════════════

YOUR OPTIMIZATION STRATEGY:
- MAPE >12%: Aggressive adstock decay adjustments (±0.1-0.15)
- MAPE 10-12%: Moderate saturation alpha tuning (±0.3-0.5)
- MAPE 8-10%: Fine-tune learning rate, regularization
- MAPE <8%: Minimal adjustments or recommend approval
- After BA rejections: Use smaller steps (half magnitude)

OUTPUT FORMAT (strict JSON):
{{
  "action": "hyperparameter_update",
  "proposed_changes": {{
    "hyperparameters": {{
      "parameter_name": new_value
    }}
  }},
  "reasoning": "Scientific explanation with formulas, expected MAPE impact",
  "flags": {{
    "needs_business_review": true/false,
    "expected_improvement_pct": 0.0
  }},
  "confidence": 0.0-1.0,
  "next_agent_suggestion": "business_analyst"
}}

Be precise, data-driven, and explain your reasoning using MMM theory."""
    
    def _build_optimization_prompt(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Dict[str, Any],
        improvement_rate: float,
        stuck_in_loop: bool,
        diminishing_returns: bool,
        ba_rejections: int
    ) -> str:
        """
        Build detailed user prompt with complete optimization context.
        
        Provides the LLM with:
        - Current performance metrics and hyperparameters
        - Full optimization history with trends
        - Convergence signals and warnings
        - Business context and benchmarks
        
        Args:
            state: Current MMMState
            history: Complete MMMHistory
            extra_context: Business benchmarks, rejection feedback
            improvement_rate: MAPE improvement rate (last 3 iterations)
            stuck_in_loop: Whether BA-DS cycle detected
            diminishing_returns: Whether improvements are declining
            ba_rejections: Number of recent Business Analyst rejections
        
        Returns:
            str: Formatted prompt for Claude
        """
        # Extract current state
        current_mape = state.performance.mape
        current_r2 = state.performance.r2
        current_mae = state.performance.mae
        current_hyperparams = state.model_config.get("hyperparameters", {})
        
        # Format recent history (last 5 iterations)
        history_summary = self._format_history_summary(history)
        
        # Add compromise mode warning if BA rejected repeatedly
        compromise_note = ""
        if ba_rejections >= 2:
            reduction = 50 * min(ba_rejections, 3)
            compromise_note = f"""
⚠️ COMPROMISE MODE: Business Analyst rejected {ba_rejections} proposals.
   → Use SMALLER adjustments (reduce magnitude by {reduction}%) to find acceptable middle ground."""
        
        # Build complete prompt
        prompt = f"""Analyze MMM performance and propose scientifically-grounded hyperparameter optimizations.

═══════════════════════════════════════════════════════════════
CURRENT MODEL STATE
═══════════════════════════════════════════════════════════════
Performance Metrics:
  • MAPE: {current_mape:.2f}% (target: <8%)
  • R²: {current_r2:.4f} (target: >0.90)
  • MAE: {current_mae:.2f}
  • Iteration: {state.iteration}

Current Hyperparameters:
{json.dumps(current_hyperparams, indent=2)}

Business Context:
  • Industry: {state.business_context.get('industry', 'Unknown')}
  • Campaign: {state.business_context.get('campaign_type', 'Unknown')}

═══════════════════════════════════════════════════════════════
OPTIMIZATION HISTORY
═══════════════════════════════════════════════════════════════
{history_summary}

Performance Trends:
  • MAPE Improvement Rate (last 3): {improvement_rate:.2f}%/iteration
  • Diminishing Returns: {"YES - Consider convergence" if diminishing_returns else "NO"}
  • Stuck in Loop: {"YES - Escalation needed" if stuck_in_loop else "NO"}
  • BA Rejections: {ba_rejections}{compromise_note}

═══════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════
As the MMM expert, analyze the above data and:

1. Identify which hyperparameter(s) to adjust based on MAPE level
2. Propose specific new values within allowed ranges
3. Estimate expected MAPE improvement (be realistic)
4. Explain using MMM theory (adstock, saturation, carryover)
5. Set confidence score based on data quality and trend clarity

REMEMBER: You can ONLY tune hyperparameters. DO NOT suggest methodology changes.

Provide response in strict JSON format as specified in system prompt."""
        
        return prompt
    
    def _format_history_summary(self, history: MMMHistory) -> str:
        """
        Format recent optimization history into readable summary.
        
        Args:
            history: Complete MMMHistory
        
        Returns:
            str: Formatted summary of last 5-8 iterations
        """
        if len(history.iterations) == 0:
            return "No previous iterations (baseline model)"
        
        recent = history.iterations[-8:]  # Last 8 iterations
        lines = []
        
        for i, record in enumerate(recent, 1):
            mape = record.state_after.performance.mape
            agent = record.agent_name
            
            # Extract what changed from structured output
            changes = record.agent_output_structured.proposed_changes if record.agent_output_structured else {}
            hyperparam_changes = changes.get("hyperparameters", {})
            change_summary = ", ".join([f"{k}→{v:.2f}" for k, v in list(hyperparam_changes.items())[:2]])
            
            lines.append(
                f"  Iter {record.iteration}: "
                f"MAPE={mape:.2f}% | "
                f"Agent={agent} | "
                f"Changes: {change_summary or 'none'}"
            )
        
        return "\n".join(lines)
    
    def _check_diminishing_returns(self, history: MMMHistory) -> bool:
        """
        Check if MAPE improvements are diminishing (convergence signal).
        
        Logic: If last 3 improvements are monotonically decreasing, returns True.
        Example: [1.2%, 0.8%, 0.4%] → True (diminishing)
        
        Args:
            history: Complete MMMHistory
        
        Returns:
            bool: True if diminishing returns detected
        """
        mape_trend = history.get_mape_trend(last_n=4)
        if len(mape_trend) < 4:
            return False
        
        # Calculate improvements between consecutive iterations
        improvements = [mape_trend[i] - mape_trend[i+1] for i in range(len(mape_trend)-1)]
        
        # Check if improvements are strictly decreasing
        if len(improvements) >= 3:
            return all(improvements[i] > improvements[i+1] for i in range(len(improvements)-1))
        
        return False
    
    def _count_recent_ba_rejections(self, history: MMMHistory) -> int:
        """
        Count Business Analyst rejections in last 5 iterations.
        
        Used to trigger "compromise mode" where smaller adjustments are made.
        
        Args:
            history: Complete MMMHistory
        
        Returns:
            int: Number of BA rejections
        """
        recent = history.iterations[-5:]
        return sum(
            1 for rec in recent
            if rec.agent_name == "business_analyst" and
               rec.agent_output_structured and
               rec.agent_output_structured.validation_results and
               rec.agent_output_structured.validation_results.get("status") in ["PARTIAL_ALIGNMENT", "CRITICAL_ISSUES"]
        )
    
    def _parse_and_validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude's JSON response and validate hyperparameter proposals.
        
        This method:
        1. Extracts JSON from markdown code blocks
        2. Parses JSON into dict
        3. Validates all proposed hyperparameters are within allowed ranges
        4. Ensures required fields exist
        5. Returns structured result dict
        
        Args:
            response_text: Raw text response from Claude API
        
        Returns:
            dict: Parsed and validated response with:
                - action: "hyperparameter_update"
                - proposed_changes: dict with hyperparameters
                - reasoning: str explanation
                - flags: dict with metadata
                - confidence: float 0.0-1.0
        
        Note:
            If parsing fails, returns safe fallback dict with empty changes.
        """
        try:
            # Extract JSON from markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse JSON
            result = json.loads(json_str)
            logger.debug(f"Successfully parsed JSON response: {json.dumps(result, indent=2)}")
            
            # Validate and constrain proposed hyperparameters
            if "proposed_changes" in result and "hyperparameters" in result["proposed_changes"]:
                validated_params = {}
                original_params = result["proposed_changes"]["hyperparameters"].copy()
                
                for param, value in result["proposed_changes"]["hyperparameters"].items():
                    if param in self.constraints:
                        # Clamp value to allowed range
                        min_val = self.constraints[param]["min"]
                        max_val = self.constraints[param]["max"]
                        clamped_value = max(min_val, min(max_val, value))
                        validated_params[param] = clamped_value
                        
                        # Log if value was clamped
                        if clamped_value != value:
                            logger.warning(
                                f"Clamped {param}: {value} → {clamped_value} (range: {min_val}-{max_val})"
                            )
                    else:
                        # Unknown parameter, skip it
                        logger.warning(f"Unknown hyperparameter '{param}' proposed, skipping")
                        result["flags"]["unknown_parameter"] = param
                
                result["proposed_changes"]["hyperparameters"] = validated_params
                logger.debug(f"Validated {len(validated_params)} hyperparameters")
            
            # Ensure required fields
            result.setdefault("action", "hyperparameter_update")
            result.setdefault("flags", {})
            result.setdefault("confidence", 0.7)
            result.setdefault("reasoning", "No reasoning provided")
            
            return result
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # Parsing failed - return safe fallback
            logger.error(f"Failed to parse LLM response: {str(e)}", exc_info=True)
            logger.debug(f"Response text that failed to parse:\n{response_text[:500]}")
            
            return {
                "action": "hyperparameter_update",
                "proposed_changes": {"hyperparameters": {}},
                "reasoning": f"Failed to parse LLM response: {str(e)}. Response: {response_text[:200]}...",
                "flags": {"parse_error": True, "error_detail": str(e)},
                "confidence": 0.0
            }
    
    def _create_error_fallback(self, error: str) -> AgentOutput:
        """
        Create safe fallback AgentOutput when API call fails.
        
        Returns a "no change" output that won't break the orchestration loop.
        
        Args:
            error: Error message from API failure
        
        Returns:
            AgentOutput: Safe fallback with no proposed changes
        """
        logger.critical(f"Creating error fallback output due to: {error}")
        
        output = AgentOutput(
            agent_name=self.name,
            action="hyperparameter_update",
            proposed_changes={"hyperparameters": {}},
            reasoning=f"Claude API error: {error}. Recommending manual review or retry.",
            flags={
                "api_error": True,
                "error_message": error,
                "requires_manual_review": True
            },
            confidence=0.0,
            next_agent_suggestion="stakeholder"
        )
        
        logger.info("Error fallback output created, routing to stakeholder for manual review")
        return output
