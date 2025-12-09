"""
Business Analyst Agent - Expert MMM Business Validator

ROLE: Business validator and strategic advisor for MMM outputs
EXPERTISE: Marketing economics, media benchmarks, ROI analysis, business constraints

This agent acts as the "Business Brain" of the MMM optimization system. It ensures that
model outputs align with real-world business knowledge, industry benchmarks, and strategic
priorities. Think of it as the seasoned marketing executive who can spot unrealistic
recommendations and protect the business from bad decisions.

CORE COMPETENCIES:
------------------
1. Marketing & Media Expertise:
   - Deep understanding of channel characteristics (TV: mass reach, brand building, long carryover)
   - Digital: performance-driven, shorter carryover, attribution challenges
   - Print: declining effectiveness, older demographics
   - OOH: location-based, brand awareness, moderate persistence
   
2. ROI & Economics Knowledge:
   - Typical ROI ranges by channel and category
   - Understanding of diminishing returns and saturation
   - Price elasticity principles (price ↑ → demand ↓)
   - Awareness of measurement biases (last-click attribution inflates digital ROI)
   
3. Business Risk Assessment:
   - Identifies recommendations that could damage brand equity
   - Flags strategic risks (e.g., cutting TV too aggressively for premium brands)
   - Considers long-term vs short-term tradeoffs
   - Understands distribution and retail partnership implications
   
4. Constraint & Guardrail Setting:
   - Proposes min/max spend limits by channel
   - Sets strategic floors (e.g., "TV must be ≥30% of budget for brand health")
   - Creates ROI caps when outputs seem unrealistic
   - Enforces business rules without touching model code

WHAT THIS AGENT DOES:
---------------------
✓ Reads model outputs (ROI curves, carryover durations, budget recommendations, coefficients)
✓ Compares outputs to industry benchmarks (by category: CPG, Auto, Retail, Finance, etc.)
✓ Flags implausible values with severity levels (warning vs critical)
✓ Proposes business constraints (in plain language, not code)
✓ Recommends data quality investigations when patterns are suspicious
✓ Translates technical results to stakeholder-friendly language
✓ Checks consistency with previously agreed strategy and constraints

WHAT THIS AGENT DOES NOT DO:
----------------------------
✗ Change methodology (protected fields: adstock type, saturation type, architecture)
✗ Edit hyperparameters directly (that's Data Scientist's job)
✗ Make final approval decisions (that's Stakeholder's job)
✗ Touch model code or training logic
✗ Force specific technical implementations

TYPICAL WORKFLOW:
----------------
1. Data Scientist proposes new model configuration
2. Model runs and produces outputs (ROI, carryover, budget recommendations)
3. Business Analyst receives outputs and analyzes them
4. BA compares to benchmarks and flags issues
5. If issues found: BA proposes constraints → back to Data Scientist
6. If all clear: BA recommends → forward to Stakeholder for approval

BENCHMARK PHILOSOPHY:
--------------------
Benchmarks are not rigid rules but guideposts. Values outside benchmarks trigger investigation,
not automatic rejection. Context matters: a tech company might have legitimately higher digital
ROI than a CPG brand. The BA's job is to flag outliers and ensure they're justified, not to
force conformity.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
from anthropic import Anthropic
from mmm_optimizer.agents.base_agent import BaseAgent, AgentOutput
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory


# ============================================================================
# INDUSTRY BENCHMARKS DATABASE
# ============================================================================
# These benchmarks are derived from industry research, meta-analyses, and
# practitioner experience. They serve as guideposts, not rigid rules.
# Values outside these ranges trigger investigation but may be justified
# by unique business contexts, data quality, or market dynamics.

INDUSTRY_BENCHMARKS = {
    "cpg": {  # Consumer Packaged Goods (food, beverage, household products)
        "roi": {
            "tv": (2.5, 4.5),        # Traditional mass reach, brand building
            "digital": (3.0, 6.0),    # Performance-driven, but attribution inflated
            "print": (1.5, 3.0),      # Declining channel, older demographics
            "ooh": (2.0, 4.0),        # Location-based, moderate effectiveness
            "radio": (2.0, 3.5),      # Audio, commuter reach
            "social": (3.5, 7.0),     # Targeted, measurable, but attribution issues
        },
        "carryover_weeks": {
            "tv": (6, 12),            # Long-lasting brand effects
            "digital": (2, 4),        # Short-term performance effects
            "print": (4, 8),          # Moderate persistence
            "ooh": (4, 10),           # Depends on exposure frequency
            "radio": (3, 6),          # Short to moderate
            "social": (1, 3),         # Very short, real-time engagement
        },
        "saturation_alpha": (1.5, 3.5),  # Hill saturation steepness
        "price_elasticity": (-0.5, -2.5), # Price sensitivity (negative)
    },
    
    "auto": {  # Automotive (cars, motorcycles, parts)
        "roi": {
            "tv": (1.5, 3.5),        # Longer purchase cycles, awareness focus
            "digital": (2.5, 5.5),    # Research-heavy, dealer lead gen
            "print": (1.0, 2.5),      # Magazines, specs, older buyers
            "ooh": (1.5, 3.0),        # Highway billboards, showroom drive-by
            "radio": (1.5, 2.5),      # Commuter radio, local dealers
            "social": (2.0, 5.0),     # Video content, engagement
        },
        "carryover_weeks": {
            "tv": (8, 16),            # Longer consideration period
            "digital": (4, 8),        # Research phase extends digital impact
            "print": (6, 12),         # Spec sheets, magazine reviews
            "ooh": (6, 12),           # Repeated exposure on commutes
            "radio": (4, 8),          # Moderate persistence
            "social": (2, 6),         # Video views, shares
        },
        "saturation_alpha": (2.0, 4.0),
        "price_elasticity": (-0.3, -1.5),  # Less price-sensitive (premium)
    },
    
    "retail": {  # E-commerce, department stores, specialty retail
        "roi": {
            "tv": (2.0, 4.0),        # Seasonal campaigns, promotions
            "digital": (4.0, 8.0),    # Direct response, e-commerce tracking
            "print": (1.0, 2.0),      # Flyers, circulars (declining)
            "ooh": (1.5, 3.0),        # Mall advertising, transit
            "radio": (1.5, 2.5),      # Local promotions
            "social": (4.5, 9.0),     # Influencer marketing, product discovery
        },
        "carryover_weeks": {
            "tv": (4, 8),             # Shorter than CPG (promotional focus)
            "digital": (1, 3),        # Very short (sale-driven)
            "print": (2, 4),          # Weekly circulars
            "ooh": (3, 6),            # Moderate
            "radio": (2, 4),          # Promotional windows
            "social": (1, 2),         # Real-time engagement
        },
        "saturation_alpha": (1.8, 3.2),
        "price_elasticity": (-1.0, -3.0),  # Highly price-sensitive
    },
    
    "finance": {  # Banking, insurance, investment services
        "roi": {
            "tv": (2.0, 4.5),        # Trust building, brand awareness
            "digital": (3.0, 7.0),    # Lead gen, calculators, forms
            "print": (1.5, 3.5),      # Financial publications, credibility
            "ooh": (1.5, 3.0),        # Airport lounges, professional visibility
            "radio": (1.5, 3.0),      # Talk radio, news formats
            "social": (2.5, 6.0),     # LinkedIn, professional content
        },
        "carryover_weeks": {
            "tv": (8, 16),            # Long consideration (mortgages, insurance)
            "digital": (4, 10),       # Research phase, comparison shopping
            "print": (6, 12),         # Magazine ads, thought leadership
            "ooh": (6, 12),           # Professional environments
            "radio": (4, 8),          # Commuter time, news context
            "social": (2, 6),         # Content marketing, lead nurturing
        },
        "saturation_alpha": (2.0, 4.0),
        "price_elasticity": (-0.2, -1.0),  # Less price-sensitive (value focus)
    },
    
    "tech": {  # Software, hardware, tech services
        "roi": {
            "tv": (1.5, 3.0),        # Brand awareness for B2C tech
            "digital": (5.0, 12.0),   # Core channel: SEO, SEM, content marketing
            "print": (0.5, 1.5),      # Minimal (tech publications only)
            "ooh": (1.0, 2.5),        # Urban markets, airports
            "radio": (1.0, 2.0),      # Podcasts, tech talk shows
            "social": (4.0, 10.0),    # Community building, product launches
        },
        "carryover_weeks": {
            "tv": (4, 8),             # Shorter cycles (product launches)
            "digital": (2, 6),        # Performance-driven, short feedback loops
            "print": (3, 6),          # Limited persistence
            "ooh": (3, 6),            # Campaign-based
            "radio": (2, 4),          # Podcast ads, short-term
            "social": (1, 4),         # Viral potential but short-lived
        },
        "saturation_alpha": (1.5, 3.0),
        "price_elasticity": (-0.5, -2.0),
    },
    
    "pharma": {  # Pharmaceuticals, healthcare products
        "roi": {
            "tv": (2.5, 5.0),        # DTC advertising (US), educational
            "digital": (3.0, 6.5),    # Symptom search, patient education
            "print": (2.0, 4.0),      # Medical journals, patient magazines
            "ooh": (1.5, 3.0),        # Pharmacy, clinic environments
            "radio": (1.5, 2.5),      # Health talk shows
            "social": (2.0, 5.0),     # Community support, awareness
        },
        "carryover_weeks": {
            "tv": (10, 20),           # Long patient journeys (diagnosis → treatment)
            "digital": (6, 12),       # Research phase, symptom management
            "print": (8, 16),         # Educational content
            "ooh": (6, 12),           # Point-of-care visibility
            "radio": (4, 8),          # Health content persistence
            "social": (4, 10),        # Support communities, ongoing engagement
        },
        "saturation_alpha": (2.5, 4.5),
        "price_elasticity": (-0.1, -0.8),  # Low (insurance coverage, necessity)
    },
}

# Default fallback if category not found
DEFAULT_BENCHMARKS = INDUSTRY_BENCHMARKS["cpg"]


class BusinessAnalystAgent(BaseAgent):
    """
    Expert Business Analyst Agent for MMM Validation
    
    This agent embodies the knowledge and judgment of a seasoned marketing executive
    with deep expertise in media mix modeling, marketing economics, and business strategy.
    
    PRIMARY RESPONSIBILITIES:
    ------------------------
    1. Validate model outputs against industry benchmarks
    2. Flag unrealistic or implausible results with clear reasoning
    3. Propose business constraints (min/max spend, strategic floors/ceilings)
    4. Translate technical outputs to stakeholder-friendly language
    5. Assess business risks of recommendations (brand equity, distribution, strategy)
    6. Recommend data quality investigations when patterns are suspicious
    
    VALIDATION FRAMEWORK:
    --------------------
    The agent performs multi-dimensional validation:
    
    - ROI Validation: Are returns realistic for each channel given category?
    - Carryover Validation: Are lag effects (adstock decay) plausible?
    - Saturation Validation: Are diminishing returns curves reasonable?
    - Budget Validation: Do recommendations respect business constraints?
    - Strategic Validation: Do outputs align with brand positioning and priorities?
    - Risk Assessment: What are the business risks of following recommendations?
    
    BENCHMARK USAGE:
    ---------------
    Benchmarks are guideposts, not rigid rules. The agent uses them to:
    - Identify outliers that need explanation
    - Calibrate expectations by category (CPG vs Tech vs Finance)
    - Trigger data quality investigations
    - Propose constraint ranges
    
    Values outside benchmarks are not automatically rejected. The agent considers:
    - Business context (premium brand vs value brand)
    - Market dynamics (competitive intensity, category growth)
    - Data quality (measurement gaps, attribution biases)
    - Strategic intent (brand building vs performance focus)
    
    CONSTRAINT PHILOSOPHY:
    ---------------------
    The agent proposes constraints in plain language, not code. Examples:
    - "TV spend must remain ≥30% of total budget for brand health"
    - "Digital ROI should be capped at 7x given attribution concerns"
    - "OOH minimum spend: $500K/quarter for market presence"
    - "Price increases limited to 5% annually to avoid elasticity risk"
    
    These constraints are recommendations for the Data Scientist to implement
    in the next optimization iteration, not direct code changes.
    """
    
    def __init__(self):
        """
        Initialize Business Analyst agent with Anthropic LLM client.
        
        Sets up:
        - Agent identity (name, description)
        - Anthropic API client for Claude
        - Benchmark database reference
        
        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable not set
        """
        super().__init__(
            name="business_analyst",
            description="Expert business validator for MMM outputs, ROI realism, and strategic constraints"
        )
        
        # Initialize Anthropic client for LLM-powered validation
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set. "
                "Get your API key from: https://console.anthropic.com/"
            )
        
        self.client = Anthropic(api_key=api_key)
        self.benchmarks = INDUSTRY_BENCHMARKS
    
    def run(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Execute business validation of current MMM state using expert LLM analysis.
        
        This method orchestrates the complete business validation workflow:
        1. Gather business context (category, brand positioning, strategic priorities)
        2. Load appropriate industry benchmarks
        3. Extract model outputs from state (hyperparameters → implied ROI/carryover)
        4. Build comprehensive validation prompt with full historical context
        5. Call Claude LLM for expert business analysis
        6. Parse validation results (issues, constraints, risk assessment)
        7. Determine next routing (back to Data Scientist or forward to Stakeholder)
        8. Apply loop prevention logic to avoid infinite BA-DS cycles
        
        Args:
            state: Current MMM state containing:
                - Hyperparameters (adstock decay, saturation alpha, etc.)
                - Performance metrics (MAPE, R², training history)
                - Business context (category, brand, market)
                - Methodology specification
            
            history: Complete optimization history containing:
                - All previous iterations
                - Agent actions and outputs
                - Performance trend over time
                - Cycle detection data
            
            extra_context: Optional additional context:
                - custom_benchmarks: Override default industry benchmarks
                - strategic_priorities: Business strategy constraints
                - existing_constraints: Previously agreed constraints
                - stakeholder_feedback: Past approval/rejection reasons
        
        Returns:
            AgentOutput containing:
                - action: "business_validation"
                - validation_results: Issues, severity, benchmarks used
                - proposed_changes: Business constraints (plain language)
                - reasoning: Detailed narrative explanation
                - flags: Critical issues, loop detection, constraint proposals
                - confidence: 0.0-1.0 based on benchmark alignment
                - next_agent_suggestion: "data_scientist" or "stakeholder"
        
        Loop Prevention:
            - Detects BA-DS cycles (repeated back-and-forth)
            - Checks for diminishing returns in performance improvements
            - Forces escalation to Stakeholder after threshold iterations
            - Accepts "best effort" state rather than infinite loops
        """
        extra_context = extra_context or {}
        
        # Step 1: Extract business context
        category = state.business_context.get("category", "cpg").lower()
        brand_positioning = state.business_context.get("brand_positioning", "mainstream")
        market = state.business_context.get("market", "us")
        
        # Step 2: Load benchmarks (custom override or industry standard)
        custom_benchmarks = extra_context.get("custom_benchmarks")
        if custom_benchmarks:
            benchmarks = custom_benchmarks
        else:
            benchmarks = self.benchmarks.get(category, DEFAULT_BENCHMARKS)
        
        # Step 3: Build validation context for LLM
        validation_context = self._build_validation_context(
            state=state,
            history=history,
            benchmarks=benchmarks,
            category=category,
            extra_context=extra_context
        )
        
        # Step 4: Call LLM for expert business analysis
        try:
            validation_result = self._call_llm_for_validation(validation_context)
        except Exception as e:
            # Fallback on LLM failure: use basic rule-based validation
            validation_result = self._fallback_validation(state, benchmarks, category)
            validation_result["reasoning"] += f" [LLM Error: {str(e)[:100]}]"
        
        # Step 5: Extract structured results
        issues = validation_result.get("issues", [])
        proposed_constraints = validation_result.get("proposed_constraints", {})
        risk_assessment = validation_result.get("risk_assessment", {})
        validation_status = validation_result.get("status", "UNKNOWN")
        reasoning = validation_result.get("reasoning", "")
        
        # Step 6: Loop prevention checks
        stuck_in_loop = history.is_stuck_in_loop(threshold=3)
        improvement_rate = history.get_mape_improvement_rate(last_n=3)
        ba_ds_cycles = history.get_agent_cycle_count(['business_analyst', 'data_scientist'])
        
        # Step 7: Determine next routing with escape logic
        next_suggestion = self._determine_next_agent(
            issues=issues,
            history_length=len(history),
            stuck_in_loop=stuck_in_loop,
            improvement_rate=improvement_rate,
            ba_ds_cycles=ba_ds_cycles
        )
        
        # Add loop escape narrative if triggered
        if next_suggestion == "stakeholder" and stuck_in_loop:
            reasoning += (
                f"\n\n[LOOP PREVENTION ACTIVATED: {ba_ds_cycles} BA-DS cycles detected. "
                f"Improvement rate: {improvement_rate:.2%}. Escalating to Stakeholder for decision.]"
            )
        elif next_suggestion == "stakeholder" and len(history) >= 12:
            reasoning += (
                f"\n\n[MAX ITERATIONS REACHED: {len(history)} iterations completed. "
                f"Accepting current state as best effort. Stakeholder review required.]"
            )
        
        # Step 8: Calculate confidence score
        confidence = self._calculate_confidence(issues, benchmarks, validation_status)
        
        # Step 9: Build flags dictionary
        flags = {
            "business_issues_found": len(issues) > 0,
            "critical_issues": any(i.get("severity") == "critical" for i in issues),
            "warning_issues": any(i.get("severity") == "warning" for i in issues),
            "proposed_constraints": proposed_constraints,
            "risk_level": risk_assessment.get("level", "LOW"),
            "stuck_in_loop": stuck_in_loop,
            "improvement_rate": improvement_rate,
            "ba_ds_cycles": ba_ds_cycles,
            "benchmarks_category": category,
        }
        
        return AgentOutput(
            agent_name=self.name,
            action="business_validation",
            proposed_changes=proposed_constraints if proposed_constraints else None,
            validation_results={
                "status": validation_status,
                "issues": issues,
                "risk_assessment": risk_assessment,
                "benchmarks_used": benchmarks,
                "category": category,
                "market": market,
                "brand_positioning": brand_positioning,
            },
            decision=None,  # BA doesn't make final decisions
            reasoning=reasoning,
            flags=flags,
            confidence=confidence,
            next_agent_suggestion=next_suggestion
        )
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_validation_context(
        self,
        state: MMMState,
        history: MMMHistory,
        benchmarks: Dict[str, Any],
        category: str,
        extra_context: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive validation context for LLM analysis.
        
        This creates a rich, structured prompt containing:
        - Current hyperparameters and implied business metrics
        - Performance trends over optimization history
        - Industry benchmarks for comparison
        - Business context (category, brand, market)
        - Strategic priorities and existing constraints
        - Historical validation issues and resolutions
        
        Args:
            state: Current MMM state
            history: Full optimization history
            benchmarks: Industry benchmarks for this category
            category: Business category (cpg, auto, retail, etc.)
            extra_context: Additional context from orchestrator
        
        Returns:
            Formatted context string for LLM prompt
        """
        # Extract current hyperparameters
        hyperparams = state.get_hyperparameters()
        
        # Calculate implied business metrics from hyperparameters
        # Note: In production, these would come from actual model outputs
        # For now, we estimate based on hyperparameter values
        implied_metrics = self._estimate_business_metrics(hyperparams)
        
        # Format performance history
        history_summary = self._format_history_for_validation(history)
        
        # Format benchmarks for display
        benchmark_summary = self._format_benchmarks(benchmarks, category)
        
        # Get strategic context
        strategic_priorities = extra_context.get("strategic_priorities", "Not specified")
        existing_constraints = extra_context.get("existing_constraints", {})
        
        context = f"""
# BUSINESS VALIDATION CONTEXT

## Current Model Configuration
Category: {category.upper()}
Brand Positioning: {state.business_context.get('brand_positioning', 'mainstream')}
Market: {state.business_context.get('market', 'US')}

## Hyperparameters
{self._format_hyperparams(hyperparams)}

## Implied Business Metrics (Estimated)
{self._format_implied_metrics(implied_metrics)}

## Performance Metrics
Current MAPE: {state.performance.get('mape', 'N/A'):.4f}
Current R²: {state.performance.get('r_squared', 'N/A'):.4f}
Iteration: {len(history) + 1}

## Optimization History Summary
{history_summary}

## Industry Benchmarks ({category.upper()})
{benchmark_summary}

## Strategic Context
Priorities: {strategic_priorities}
Existing Constraints: {existing_constraints if existing_constraints else 'None'}

## Previous Validation Issues
{self._format_previous_issues(history)}
"""
        return context
    
    def _call_llm_for_validation(self, context: str) -> Dict[str, Any]:
        """
        Call Claude LLM for expert business validation analysis.
        
        Sends a structured prompt to Claude asking it to:
        1. Compare current metrics to industry benchmarks
        2. Identify implausible or concerning values
        3. Assess business risks of current configuration
        4. Propose constraints to address issues
        5. Provide clear reasoning and severity levels
        
        Args:
            context: Rich validation context string
        
        Returns:
            Dictionary containing:
                - status: "FULL_ALIGNMENT" | "PARTIAL_ALIGNMENT" | "CRITICAL_ISSUES"
                - issues: List of validation issues with severity
                - proposed_constraints: Plain language constraints
                - risk_assessment: Business risk analysis
                - reasoning: Detailed narrative explanation
        """
        system_prompt = """You are an expert Marketing Mix Modeling (MMM) business analyst with 15+ years of experience in media analytics, marketing economics, and business strategy.

Your expertise includes:
- Deep knowledge of channel characteristics (TV: mass reach, brand building; Digital: performance-driven; Print: declining; OOH: location-based)
- Industry ROI benchmarks across categories (CPG, Auto, Retail, Finance, Tech, Pharma)
- Understanding of adstock/carryover effects and saturation dynamics
- Marketing economics principles (price elasticity, diminishing returns)
- Business risk assessment (brand equity, distribution, strategic alignment)
- Data quality red flags (attribution biases, measurement gaps)

Your role is to:
1. Validate model outputs against industry benchmarks
2. Flag unrealistic or implausible values with clear severity levels
3. Consider business context (category, brand positioning, market dynamics)
4. Propose business constraints (NOT code changes) to address issues
5. Assess strategic risks of recommendations
6. Recommend data quality investigations when patterns are suspicious

CRITICAL RULES:
- You CANNOT change methodology (adstock type, saturation type, architecture)
- You CANNOT edit hyperparameters directly (that's Data Scientist's job)
- You CAN propose constraints in plain language (e.g., "TV ROI cap at 4.5x")
- Benchmarks are guideposts, not rigid rules - context matters
- Flag outliers but allow justification (tech companies may have higher digital ROI)

Output strict JSON format:
{
  "status": "FULL_ALIGNMENT" | "PARTIAL_ALIGNMENT" | "CRITICAL_ISSUES",
  "issues": [
    {
      "channel": "TV" | "Digital" | "Print" | "OOH" | "Other",
      "metric": "ROI" | "Carryover" | "Saturation" | "Budget",
      "value": "actual value",
      "benchmark": "expected range",
      "severity": "warning" | "critical",
      "reasoning": "why this is concerning",
      "recommendation": "what to do about it"
    }
  ],
  "proposed_constraints": {
    "constraint_name": "plain language constraint description"
  },
  "risk_assessment": {
    "level": "LOW" | "MEDIUM" | "HIGH",
    "concerns": ["list of business risks"],
    "mitigation": "recommendations"
  },
  "reasoning": "comprehensive narrative explanation"
}"""

        user_prompt = f"""{context}

Based on this context, perform a comprehensive business validation:

1. **ROI Validation**: Are the implied ROI values realistic for this category?
2. **Carryover Validation**: Are the adstock decay rates (implied carryover durations) plausible?
3. **Saturation Validation**: Are the saturation parameters reasonable?
4. **Strategic Alignment**: Do outputs align with business positioning and priorities?
5. **Risk Assessment**: What are the business risks of these recommendations?
6. **Data Quality**: Are there suspicious patterns suggesting measurement issues?

Return your analysis in strict JSON format as specified."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for consistent, analytical output
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse JSON response
        response_text = response.content[0].text.strip()
        
        # Handle markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        return json.loads(response_text)
    
    def _fallback_validation(
        self,
        state: MMMState,
        benchmarks: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """
        Fallback validation using rule-based logic when LLM fails.
        
        This provides basic validation by:
        1. Estimating ROI/carryover from hyperparameters
        2. Comparing to benchmark ranges
        3. Flagging critical deviations
        
        Not as sophisticated as LLM analysis but ensures validation
        never completely fails.
        
        Args:
            state: Current MMM state
            benchmarks: Industry benchmarks
            category: Business category
        
        Returns:
            Validation results in same format as LLM output
        """
        hyperparams = state.get_hyperparameters()
        implied_metrics = self._estimate_business_metrics(hyperparams)
        
        issues = []
        
        # Check TV ROI
        tv_roi = implied_metrics.get("tv_roi")
        tv_roi_range = benchmarks.get("roi", {}).get("tv", (2.5, 4.5))
        if tv_roi and (tv_roi < tv_roi_range[0] or tv_roi > tv_roi_range[1]):
            deviation = max(tv_roi_range[0] - tv_roi, tv_roi - tv_roi_range[1])
            severity = "critical" if deviation > 1.0 else "warning"
            issues.append({
                "channel": "TV",
                "metric": "ROI",
                "value": f"{tv_roi:.1f}x",
                "benchmark": f"{tv_roi_range[0]}-{tv_roi_range[1]}x",
                "severity": severity,
                "reasoning": f"TV ROI outside expected range for {category}",
                "recommendation": "Review TV adstock decay and saturation parameters"
            })
        
        # Check Digital ROI
        digital_roi = implied_metrics.get("digital_roi")
        digital_roi_range = benchmarks.get("roi", {}).get("digital", (3.0, 6.0))
        if digital_roi and (digital_roi < digital_roi_range[0] or digital_roi > digital_roi_range[1]):
            deviation = max(digital_roi_range[0] - digital_roi, digital_roi - digital_roi_range[1])
            severity = "critical" if deviation > 2.0 else "warning"
            issues.append({
                "channel": "Digital",
                "metric": "ROI",
                "value": f"{digital_roi:.1f}x",
                "benchmark": f"{digital_roi_range[0]}-{digital_roi_range[1]}x",
                "severity": severity,
                "reasoning": f"Digital ROI outside expected range (attribution inflation risk)",
                "recommendation": "Investigate digital attribution data quality"
            })
        
        # Determine status
        if not issues:
            status = "FULL_ALIGNMENT"
        elif any(i["severity"] == "critical" for i in issues):
            status = "CRITICAL_ISSUES"
        else:
            status = "PARTIAL_ALIGNMENT"
        
        # Propose constraints
        proposed_constraints = {}
        if digital_roi and digital_roi > digital_roi_range[1]:
            proposed_constraints["digital_roi_cap"] = f"Cap digital ROI at {digital_roi_range[1]}x"
        
        return {
            "status": status,
            "issues": issues,
            "proposed_constraints": proposed_constraints,
            "risk_assessment": {
                "level": "MEDIUM" if issues else "LOW",
                "concerns": [i["reasoning"] for i in issues],
                "mitigation": "Address flagged issues via Data Scientist"
            },
            "reasoning": f"Rule-based validation for {category}. {len(issues)} issues found."
        }
    
    def _estimate_business_metrics(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate business metrics (ROI, carryover) from hyperparameters.
        
        This is a rough estimation function used when actual model outputs
        aren't available. In production, you'd use real model predictions.
        
        Estimation logic:
        - Higher adstock decay → longer carryover → typically higher ROI
        - Higher saturation alpha → steeper diminishing returns → lower peak ROI
        - Learning rate affects convergence, not business metrics directly
        
        Args:
            hyperparams: Current hyperparameter configuration
        
        Returns:
            Dictionary of estimated metrics:
                - tv_roi, digital_roi, print_roi, ooh_roi
                - tv_carryover_weeks, digital_carryover_weeks
                - saturation_steepness
        """
        # Adstock decay → carryover duration (weeks)
        # Formula: effective_weeks ≈ -ln(0.1) / ln(decay)
        # (time until effect decays to 10% of original)
        import math
        
        tv_decay = hyperparams.get("tv_adstock_decay", 0.5)
        digital_decay = hyperparams.get("digital_adstock_decay", 0.4)
        print_decay = hyperparams.get("print_adstock_decay", 0.3)
        ooh_decay = hyperparams.get("ooh_adstock_decay", 0.5)
        
        tv_carryover = -math.log(0.1) / -math.log(tv_decay) if tv_decay < 1 else 10
        digital_carryover = -math.log(0.1) / -math.log(digital_decay) if digital_decay < 1 else 5
        
        # Saturation alpha affects ROI ceiling
        # Higher alpha → steeper curve → lower sustainable ROI
        saturation_alpha = hyperparams.get("saturation_alpha", 2.5)
        saturation_factor = 1.0 - (saturation_alpha - 1.5) / 5.0  # Normalize
        
        # Rough ROI estimation (baseline + decay boost × saturation adjustment)
        tv_roi = (2.5 + (tv_decay * 3.0)) * saturation_factor
        digital_roi = (3.0 + (digital_decay * 5.0)) * saturation_factor
        print_roi = (1.5 + (print_decay * 2.0)) * saturation_factor
        ooh_roi = (2.0 + (ooh_decay * 3.0)) * saturation_factor
        
        return {
            "tv_roi": round(tv_roi, 2),
            "digital_roi": round(digital_roi, 2),
            "print_roi": round(print_roi, 2),
            "ooh_roi": round(ooh_roi, 2),
            "tv_carryover_weeks": round(tv_carryover, 1),
            "digital_carryover_weeks": round(digital_carryover, 1),
            "saturation_steepness": saturation_alpha,
        }
    
    def _format_hyperparams(self, hyperparams: Dict[str, Any]) -> str:
        """Format hyperparameters for LLM display."""
        lines = []
        for key, value in hyperparams.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)
    
    def _format_implied_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format implied business metrics for LLM display."""
        return f"""  ROI Estimates:
    - TV: {metrics.get('tv_roi', 'N/A')}x
    - Digital: {metrics.get('digital_roi', 'N/A')}x
    - Print: {metrics.get('print_roi', 'N/A')}x
    - OOH: {metrics.get('ooh_roi', 'N/A')}x
  
  Carryover Duration Estimates:
    - TV: {metrics.get('tv_carryover_weeks', 'N/A')} weeks
    - Digital: {metrics.get('digital_carryover_weeks', 'N/A')} weeks
  
  Saturation Steepness: {metrics.get('saturation_steepness', 'N/A')}"""
    
    def _format_history_for_validation(self, history: MMMHistory) -> str:
        """Format optimization history for LLM context."""
        if len(history) == 0:
            return "  No previous iterations (first run)"
        
        # Get last 5 iterations
        recent = history.get_recent(n=5)
        
        lines = []
        for i, record in enumerate(recent, 1):
            mape = record.output.validation_results.get("mape") if record.output.validation_results else None
            agent = record.agent_name
            action = record.output.action
            lines.append(f"  Iteration -{len(recent)-i+1}: Agent={agent}, Action={action}, MAPE={mape}")
        
        # Add trend
        improvement = history.get_mape_improvement_rate(last_n=5)
        lines.append(f"\n  Improvement Rate (last 5): {improvement:.2%}")
        
        return "\n".join(lines)
    
    def _format_benchmarks(self, benchmarks: Dict[str, Any], category: str) -> str:
        """Format benchmarks for LLM display."""
        roi = benchmarks.get("roi", {})
        carryover = benchmarks.get("carryover_weeks", {})
        
        lines = [f"  Category: {category.upper()}\n"]
        lines.append("  ROI Ranges:")
        for channel, range_tuple in roi.items():
            lines.append(f"    - {channel.title()}: {range_tuple[0]}-{range_tuple[1]}x")
        
        lines.append("\n  Carryover Duration (weeks):")
        for channel, range_tuple in carryover.items():
            lines.append(f"    - {channel.title()}: {range_tuple[0]}-{range_tuple[1]} weeks")
        
        return "\n".join(lines)
    
    def _format_previous_issues(self, history: MMMHistory) -> str:
        """Extract and format previous validation issues from history."""
        if len(history) == 0:
            return "  No previous validation issues"
        
        # Look through history for BA outputs with issues
        issues_found = []
        for record in history.get_recent(n=5):
            if record.agent_name == "business_analyst":
                validation = record.output.validation_results
                if validation and validation.get("issues"):
                    for issue in validation["issues"]:
                        issues_found.append(
                            f"  - {issue.get('channel')} {issue.get('metric')}: {issue.get('reasoning')}"
                        )
        
        if not issues_found:
            return "  No significant issues in recent history"
        
        return "\n".join(issues_found[:3])  # Limit to 3 most recent
    
    def _determine_next_agent(
        self,
        issues: List[Dict[str, Any]],
        history_length: int,
        stuck_in_loop: bool,
        improvement_rate: float,
        ba_ds_cycles: int
    ) -> str:
        """
        Determine which agent should act next based on validation results and loop prevention.
        
        Routing logic:
        1. If stuck in loop OR diminishing returns → Stakeholder (force decision)
        2. If critical issues AND under iteration limit → Data Scientist (fix)
        3. If no issues → Stakeholder (ready for approval)
        4. If max iterations reached with issues → Stakeholder (best effort)
        
        Args:
            issues: List of validation issues
            history_length: Number of iterations completed
            stuck_in_loop: Boolean flag for BA-DS cycle detection
            improvement_rate: Recent MAPE improvement rate
            ba_ds_cycles: Count of BA→DS→BA cycles
        
        Returns:
            "data_scientist" or "stakeholder"
        """
        # ESCAPE ROUTE 1: Stuck in loop
        if stuck_in_loop or ba_ds_cycles >= 4:
            return "stakeholder"
        
        # ESCAPE ROUTE 2: Diminishing returns with many iterations
        if improvement_rate < 0.05 and history_length >= 8:
            return "stakeholder"
        
        # ESCAPE ROUTE 3: Max iterations reached
        if history_length >= 12:
            return "stakeholder"
        
        # Normal routing: issues → DS, no issues → Stakeholder
        if len(issues) > 0:
            return "data_scientist"
        else:
            return "stakeholder"
    
    def _calculate_confidence(
        self,
        issues: List[Dict[str, Any]],
        benchmarks: Dict[str, Any],
        validation_status: str
    ) -> float:
        """
        Calculate confidence score for validation.
        
        Confidence is based on:
        - Number and severity of issues
        - Alignment with benchmarks
        - Validation status
        
        Args:
            issues: List of validation issues
            benchmarks: Industry benchmarks used
            validation_status: Overall validation status
        
        Returns:
            Confidence score 0.0-1.0
        """
        if validation_status == "FULL_ALIGNMENT":
            return 0.95
        elif validation_status == "PARTIAL_ALIGNMENT":
            # Reduce confidence based on number and severity
            critical_count = sum(1 for i in issues if i.get("severity") == "critical")
            warning_count = len(issues) - critical_count
            
            penalty = (critical_count * 0.15) + (warning_count * 0.08)
            return max(0.5, 0.85 - penalty)
        else:  # CRITICAL_ISSUES
            return max(0.3, 0.6 - (len(issues) * 0.05))
