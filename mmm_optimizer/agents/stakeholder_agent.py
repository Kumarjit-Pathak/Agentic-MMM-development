"""
Stakeholder Agent - Executive Decision Maker for MMM Deployment

ROLE: Chief Marketing Officer / Senior Marketing Leadership (FMCG/CPG Context)
AUTHORITY: Final approval/rejection of MMM recommendations and budget reallocations

This agent embodies the strategic perspective and decision-making authority of a seasoned
C-suite marketing executive in the Fast-Moving Consumer Goods (FMCG) / Consumer Packaged
Goods (CPG) industry. It makes the final "go/no-go" decision on deploying optimized MMM
recommendations, balancing short-term sales lift with long-term brand equity.

EXECUTIVE MINDSET:
------------------
The Stakeholder thinks like a CMO who must answer to the CEO, CFO, and Board. Key concerns:

1. **Sales & Financial Performance**
   - Will this drive incremental sales? By how much?
   - What's the ROI improvement vs current allocation?
   - Can we afford the transition costs?
   - What's the payback period?

2. **Brand Equity & Long-Term Health**
   - TV and sponsorships build brand, not just sales
   - Cutting brand-building channels too aggressively risks future erosion
   - Premium brands need sustained visibility
   - Distribution partners expect consistent media support

3. **Strategic Alignment**
   - Does this fit our brand strategy (growth vs defend, premium vs mass)?
   - Are we maintaining competitive share-of-voice?
   - Does this support new product launches or market expansion?
   - Are we protecting strategic channel relationships?

4. **Risk Management**
   - Execution risk: Can we implement these changes?
   - Brand risk: Will awareness or perception suffer?
   - Financial risk: What if the model is wrong?
   - Organizational risk: Can the team manage this transition?

5. **Organizational Readiness**
   - Do we have the digital capabilities for increased spend?
   - Are TV contracts flexible enough for cuts?
   - Can retail partnerships handle shifts in trade support?
   - Is the team equipped to execute and monitor?

DECISION FRAMEWORK:
-------------------
The Stakeholder reviews a comprehensive executive summary including:

**Model Performance:**
- Final MAPE (Mean Absolute Percentage Error) - lower is better
- R² (coefficient of determination) - explains variance, higher is better
- Model stability and convergence (# iterations, trajectory)
- Out-of-sample validation performance

**Recommended Budget Reallocation:**
- Current allocation by channel (TV, Digital, Print, OOH, etc.)
- Recommended allocation with % changes
- Rationale for each shift (diminishing returns, underinvestment, etc.)

**Expected Business Impact:**
- Sales lift projection (e.g., +3.5% incremental sales)
- ROI improvement by channel
- Total marketing efficiency gain
- Contribution to business objectives (volume, revenue, profit)

**Agent Validations:**
- Data Scientist: Model quality, convergence, hyperparameter soundness
- Business Analyst: ROI realism, benchmark alignment, constraint compliance
- Research Persona: Methodology integrity (zero violations acceptable)

**Business Context:**
- Brand positioning (premium vs value, growth vs defend)
- Category dynamics (competitive intensity, seasonality, trends)
- Market/region specifics (US vs Europe vs Asia, urban vs rural)
- Organizational constraints (contracts, capabilities, politics)

**Risk Assessment:**
- Financial risk (investment size, ROI uncertainty)
- Brand risk (awareness/perception impact of media shifts)
- Execution risk (organizational capability to deliver)
- Measurement risk (attribution confidence, data quality)

DECISION OPTIONS:
-----------------
1. **APPROVED**
   - Full endorsement, ready for immediate national rollout
   - High confidence in model, recommendations align with strategy
   - Low execution risk, all validations passed cleanly
   - Example: "Minor optimizations within existing strategy"

2. **CONDITIONAL_APPROVAL**
   - Approved with guardrails, phased rollout, or monitoring requirements
   - Model quality good, but changes are significant or risky
   - Need proof points before full commitment
   - Example: "Approve but phase TV cuts over 2 quarters, monitor brand weekly"

3. **APPROVED_FOR_TESTING**
   - Pilot/test in limited geography or time period first
   - Promising results but unproven at scale
   - Want real-world validation before full rollout
   - Example: "Test in 2-3 markets for 8 weeks, then decide on national scale"

4. **REJECTED**
   - Not ready for deployment, need more work
   - Critical unresolved issues (data quality, methodology violations, unrealistic ROI)
   - Misalignment with strategy or unacceptable risk
   - Example: "Digital ROI assumptions implausible, TV cuts too aggressive for premium brand"

OUTPUT DELIVERABLES:
--------------------
For each decision, the Stakeholder provides:

**Decision**: One of the above four options

**Reasoning**: Business narrative (no technical jargon) explaining:
- Why this decision was made
- What factors were most important
- What risks were considered
- What gives confidence (or concern)

**Implementation Plan**: Tactical roadmap including:
- Phasing: Q1/Q2/Q3 timeline, which channels change when
- Geography: National vs regional rollout
- Budget transition: Immediate vs gradual shifts
- Team readiness: Training, tools, processes needed

**Success Criteria**: Clear measurable targets:
- Sales lift target (e.g., "≥2.5% incremental sales within 6 months")
- ROI improvement (e.g., "Digital ROI >5x, TV ROI 3-4x")
- Brand health (e.g., "Aided awareness ≥85%, consideration ≥60%")
- Market share maintenance (e.g., "Hold ≥20% category share")

**Risk Mitigation & Monitoring**:
- What to track (brand metrics, sales velocity, attribution quality)
- How often (daily, weekly, monthly)
- Red flags that trigger intervention (e.g., "If brand awareness drops >5 pts, pause TV cuts")
- Rollback plan if things go wrong

**Next Steps**: Optional follow-ups for team:
- Additional analysis needed (e.g., "Simulate scenario with -15% TV instead of -25%")
- Preparatory work (e.g., "Set up brand tracking survey, negotiate TV contract flexibility")
- Stakeholder engagement (e.g., "Present to Board, get CEO signoff on digital investment")

WHAT THIS AGENT DOES:
----------------------
✓ Reviews complete MMM optimization executive summary
✓ Evaluates model quality and recommendation soundness
✓ Assesses strategic fit with brand positioning and business goals
✓ Weighs risks (financial, brand, execution, measurement)
✓ Makes final go/no-go decision with clear reasoning
✓ Defines implementation plan, success criteria, and monitoring
✓ Sets guardrails and conditions for safe deployment

WHAT THIS AGENT DOES NOT DO:
-----------------------------
✗ Tune hyperparameters or change model configuration (Data Scientist's job)
✗ Validate ROI or benchmark alignment (Business Analyst's job)
✗ Check methodology compliance (Research Persona's job)
✗ Override technical or business validation findings
✗ Force specific technical implementations
✗ Change protected methodology fields

DECISION PRINCIPLES:
--------------------
These principles guide the agent's judgment:

**Never Approve If:**
- Research Persona flagged methodology violations (zero tolerance)
- Business Analyst has unresolved CRITICAL flags (data quality, implausible ROI)
- Model performance is poor (MAPE >10%, R² <0.7)
- Recommendations conflict fundamentally with brand strategy

**Lean Towards CONDITIONAL_APPROVAL When:**
- Model quality is good (MAPE <8-9%, R² >0.85)
- Recommendations involve significant shifts (e.g., TV -20%, Digital +40%)
- Changes are directionally right but magnitude is aggressive
- Want to "test the waters" with monitoring before full commitment

**Use APPROVED_FOR_TESTING When:**
- Results look promising but unproven (new market, new methodology)
- Stakeholder wants real-world validation before national scale
- Changes are substantial and reversibility is important
- Organizational learning is needed before full rollout

**Use REJECTED When:**
- Critical unresolved issues in validation (DS, BA, or RP flags)
- Recommendations are unrealistic or implausible given market knowledge
- Execution risk is too high (org not ready, contracts inflexible)
- Conflicts with strategic priorities (e.g., cutting TV during brand relaunch)

**Brand Equity Lens:**
- Premium brands: Protect TV/brand-building channels, phase cuts slowly
- Growth brands: Accept higher risk for performance channels if proven
- Mature brands: Balance efficiency (cut waste) with maintenance (sustain awareness)
- Defend brands: Conservative changes, focus on proven channels

This agent brings the "adult in the room" perspective, ensuring that analytically optimal
recommendations are also strategically sound and operationally feasible.
"""

from typing import Dict, Any, Optional, List
import os
import json
from anthropic import Anthropic
from mmm_optimizer.agents.base_agent import BaseAgent, AgentOutput
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory


class StakeholderAgent(BaseAgent):
    """
    Executive Stakeholder Agent - CMO-Level Strategic Decision Maker
    
    This agent embodies the strategic judgment and decision-making authority of a Chief
    Marketing Officer (CMO) or senior marketing leadership in the FMCG/CPG industry.
    
    ROLE & AUTHORITY:
    ----------------
    - Final approval/rejection authority for MMM recommendations
    - Balances short-term sales optimization with long-term brand equity
    - Weighs analytical rigor against business pragmatism and risk
    - Ensures organizational readiness for recommended changes
    - Protects brand health and strategic positioning
    
    STRATEGIC PERSPECTIVE:
    ---------------------
    Thinks like a CMO accountable to CEO, CFO, and Board, considering:
    
    **Financial Performance:**
    - Incremental sales lift and ROI improvement
    - Marketing efficiency gains and budget optimization
    - Payback period and investment risk
    
    **Brand Equity:**
    - TV/sponsorships build brand, not just immediate sales
    - Premium brands need sustained visibility for long-term equity
    - Distribution partners expect consistent media support
    - Aggressive cuts to brand-building channels risk future erosion
    
    **Strategic Alignment:**
    - Does this support brand strategy (growth/defend, premium/mass)?
    - Maintains competitive share-of-voice?
    - Aligns with product launches, market expansion plans?
    - Protects strategic channel relationships?
    
    **Risk Management:**
    - Execution: Can we implement these changes?
    - Brand: Will awareness/perception suffer?
    - Financial: What if model predictions are wrong?
    - Organizational: Is the team ready?
    
    DECISION FRAMEWORK:
    ------------------
    Reviews comprehensive executive summary:
    1. Model performance (MAPE, R², stability, convergence)
    2. Budget reallocation recommendations (current vs proposed)
    3. Expected impact (sales lift, ROI improvement, KPIs)
    4. Agent validations (Data Scientist, Business Analyst, Research Persona)
    5. Business context (brand positioning, market dynamics, constraints)
    6. Risk assessment (financial, brand, execution, measurement)
    
    Makes one of four decisions:
    - **APPROVED**: Full national rollout, high confidence
    - **CONDITIONAL_APPROVAL**: Approved with guardrails, phasing, monitoring
    - **APPROVED_FOR_TESTING**: Geo pilot first, then decide on scale
    - **REJECTED**: Not ready, needs more work
    
    DECISION PRINCIPLES:
    -------------------
    - Never approve if Research Persona flagged methodology violations
    - Never approve if Business Analyst has unresolved CRITICAL flags
    - Lean towards phased rollout for significant budget shifts (>20%)
    - Protect brand-building channels for premium brands
    - Require stronger evidence for aggressive digital bets (attribution risk)
    - Test before scaling for unproven strategies
    - Accept "good enough" over "perfect" if improvement rate is diminishing
    """
    
    # Decision thresholds (guidelines, not rigid rules)
    MAPE_EXCELLENT = 8.0       # Outstanding model quality
    MAPE_GOOD = 10.0           # Acceptable for approval
    MAPE_ACCEPTABLE = 12.0     # May approve for testing only
    R2_STRONG = 0.85           # Strong explanatory power
    R2_ACCEPTABLE = 0.75       # Minimum acceptable
    MAX_ITERATIONS = 15        # Preferred iteration limit
    
    # Budget shift thresholds for phasing decisions
    MODERATE_SHIFT = 0.15      # 15% change triggers caution
    MAJOR_SHIFT = 0.25         # 25% change requires phasing
    
    def __init__(self):
        """
        Initialize Stakeholder agent with Anthropic LLM client.
        
        Sets up:
        - Agent identity (name, description)
        - Anthropic API client for Claude (CMO-level strategic reasoning)
        - Decision thresholds
        
        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable not set
        """
        super().__init__(
            name="stakeholder",
            description="Executive CMO-level approver for MMM deployment decisions"
        )
        
        # Initialize Anthropic client for LLM-powered strategic decision making
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set. "
                "Get your API key from: https://console.anthropic.com/"
            )
        
        self.client = Anthropic(api_key=api_key)
    
    def run(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Execute executive review and make final deployment decision.
        
        This method orchestrates the complete stakeholder decision-making workflow:
        1. Build executive summary (model performance, optimization trajectory)
        2. Extract budget recommendations (current vs proposed allocation)
        3. Calculate expected business impact (sales lift, ROI improvement)
        4. Review agent validations (DS, BA, RP status)
        5. Assess strategic fit and risk (brand, execution, financial)
        6. Call Claude LLM for CMO-level strategic decision
        7. Parse decision (APPROVED, CONDITIONAL_APPROVAL, APPROVED_FOR_TESTING, REJECTED)
        8. Define implementation plan, success criteria, monitoring requirements
        9. Set next steps if additional work is needed
        
        DECISION LOGIC:
        --------------
        The agent applies CMO-level judgment considering:
        
        **Auto-Reject Conditions:**
        - Research Persona flagged methodology violations (zero tolerance)
        - Business Analyst has unresolved CRITICAL issues
        - Model quality is poor (MAPE >12%, R² <0.70)
        - Recommendations fundamentally conflict with brand strategy
        
        **Conditional Approval Triggers:**
        - Budget shifts >15% (moderate) or >25% (major) in any channel
        - TV cuts for premium brands (brand equity risk)
        - Large digital increases (attribution confidence concerns)
        - Good model quality but organizational readiness concerns
        
        **Testing-Only Triggers:**
        - Promising results but unproven strategy
        - Substantial changes with reversibility concerns
        - Want real-world validation before full scale
        
        **Full Approval Conditions:**
        - Excellent model quality (MAPE <8%, R² >0.85)
        - All validations passed cleanly
        - Recommendations are moderate and aligned with strategy
        - Low execution risk, high confidence
        
        Args:
            state: Current MMM state containing:
                - Model performance (MAPE, R², training history)
                - Hyperparameters and configuration
                - Business context (category, brand, market)
                - Current budget allocation
            
            history: Complete optimization history containing:
                - All previous iterations
                - Agent actions and validations
                - Performance trajectory
                - Methodology violation records
            
            extra_context: Executive briefing materials:
                - executive_summary: Pre-formatted summary (optional)
                - recommended_allocation: Budget reallocation table
                - expected_impact: Sales lift, ROI projections
                - ds_validation: Data Scientist assessment
                - ba_validation: Business Analyst assessment
                - rp_validation: Research Persona assessment
                - strategic_priorities: Business strategy context
                - market_conditions: External factors, competitive dynamics
                - organizational_readiness: Capability assessment
                - risk_assessment: Risk analysis summary
        
        Returns:
            AgentOutput containing:
                - action: "approval_decision"
                - decision: "APPROVED" | "CONDITIONAL_APPROVAL" | "APPROVED_FOR_TESTING" | "REJECTED"
                - reasoning: Business narrative explaining decision
                - flags: Implementation plan, success criteria, risk mitigation
                - proposed_changes: Conditions or requirements (if conditional/rejected)
                - confidence: 0.0-1.0 based on validation alignment and risk
                - next_agent_suggestion: None (terminal) or agent name if more work needed
        
        Example Decisions:
        -----------------
        **APPROVED:**
        "Model quality excellent (MAPE 7.2%, R² 0.89). Recommendations are moderate
        optimizations within existing strategy. All validations passed. Ready for
        immediate national rollout with standard monitoring."
        
        **CONDITIONAL_APPROVAL:**
        "Model quality good (MAPE 8.5%, R² 0.86) but TV reduction of -22% is aggressive
        for premium brand. Approve with phased implementation: -11% Q1, -11% Q2. Monitor
        brand awareness weekly. If awareness drops >5pts, pause further cuts."
        
        **APPROVED_FOR_TESTING:**
        "Promising results (MAPE 7.8%, R² 0.87) but Digital increase of +45% is unproven
        at scale. Approve geo test in 3 markets for 8 weeks. Measure incremental sales
        lift. If successful (>2.5% lift), proceed to national rollout."
        
        **REJECTED:**
        "Business Analyst flagged critical issues: Digital ROI of 9.2x exceeds plausible
        range (3-6x), suggesting attribution inflation. TV cuts of -35% conflict with
        brand relaunch strategy. Reject pending data quality review and strategy alignment."
        """
        extra_context = extra_context or {}
        
        # Step 1: Build comprehensive executive summary
        exec_summary = self._build_executive_summary(state, history, extra_context)
        
        # Step 2: Extract validation status from history
        validation_status = self._extract_validation_status(history)
        
        # Step 3: Check for auto-reject conditions
        auto_reject_reason = self._check_auto_reject_conditions(
            state, history, validation_status
        )
        
        if auto_reject_reason:
            # Auto-reject without LLM call (clear violation of hard constraints)
            return self._create_rejection_output(auto_reject_reason, state, history)
        
        # Step 4: Call LLM for strategic decision
        try:
            decision_result = self._call_llm_for_decision(exec_summary, validation_status)
        except Exception as e:
            # Fallback on LLM failure: use conservative rule-based decision
            decision_result = self._fallback_decision(state, history, validation_status)
            decision_result["reasoning"] += f" [LLM Error: {str(e)[:100]}]"
        
        # Step 5: Extract decision components
        decision = decision_result.get("decision", "REJECTED")
        reasoning = decision_result.get("reasoning", "")
        implementation_plan = decision_result.get("implementation_plan", {})
        success_criteria = decision_result.get("success_criteria", [])
        risk_mitigation = decision_result.get("risk_mitigation", {})
        conditions = decision_result.get("conditions", [])
        next_steps = decision_result.get("next_steps", [])
        
        # Step 6: Determine if more work is needed
        next_agent_suggestion = None
        if decision == "REJECTED" and next_steps:
            # Suggest which agent should address the issues
            if any("data quality" in step.lower() or "model" in step.lower() for step in next_steps):
                next_agent_suggestion = "data_scientist"
            elif any("benchmark" in step.lower() or "business" in step.lower() for step in next_steps):
                next_agent_suggestion = "business_analyst"
        
        # Step 7: Calculate confidence
        confidence = self._calculate_decision_confidence(
            decision, validation_status, state, history
        )
        
        # Step 8: Build flags dictionary
        flags = {
            "decision_type": decision,
            "implementation_plan": implementation_plan,
            "success_criteria": success_criteria,
            "risk_mitigation": risk_mitigation,
            "validation_summary": validation_status,
            "model_quality": {
                "mape": state.performance.get("mape"),
                "r_squared": state.performance.get("r_squared"),
                "iterations": len(history),
            }
        }
        
        # Add conditions for conditional approval
        if decision == "CONDITIONAL_APPROVAL" and conditions:
            flags["conditions"] = conditions
        
        # Add requirements for rejection
        if decision == "REJECTED" and next_steps:
            flags["required_improvements"] = next_steps
        
        # Step 9: Map decision to action
        action_map = {
            "APPROVED": "full_approval",
            "CONDITIONAL_APPROVAL": "conditional_approval",
            "APPROVED_FOR_TESTING": "test_approval",
            "REJECTED": "rejection"
        }
        action = action_map.get(decision, "rejection")
        
        return AgentOutput(
            agent_name=self.name,
            action=action,
            proposed_changes=conditions if conditions else None,
            validation_results=None,  # Stakeholder doesn't validate, only decides
            decision=decision,
            reasoning=reasoning,
            flags=flags,
            confidence=confidence,
            next_agent_suggestion=next_agent_suggestion
        )
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_executive_summary(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive executive summary for CMO-level decision making.
        
        This creates a rich, business-focused summary containing:
        - Model performance and optimization trajectory
        - Budget reallocation recommendations
        - Expected business impact (sales lift, ROI)
        - Agent validation status
        - Business context and strategic considerations
        - Risk assessment
        
        Args:
            state: Current MMM state
            history: Full optimization history
            extra_context: Additional context (strategy, market, risk)
        
        Returns:
            Formatted executive summary for LLM prompt
        """
        # Performance metrics
        final_mape = state.performance.get("mape", "N/A")
        final_r2 = state.performance.get("r_squared", "N/A")
        total_iterations = len(history)
        
        # Performance trajectory
        if len(history) >= 3:
            recent_mapes = [
                r.output.validation_results.get("mape")
                for r in history.get_recent(3)
                if r.output.validation_results and r.output.validation_results.get("mape")
            ]
            if recent_mapes:
                improvement = ((recent_mapes[0] - recent_mapes[-1]) / recent_mapes[0] * 100)
                trajectory = f"Improving ({improvement:+.1f}% over last 3 iterations)"
            else:
                trajectory = "Stable"
        else:
            trajectory = "Initial iterations"
        
        # Business context
        category = state.business_context.get("category", "cpg").upper()
        brand_positioning = state.business_context.get("brand_positioning", "mainstream")
        market = state.business_context.get("market", "US")
        
        # Strategic priorities from context
        strategic_priorities = extra_context.get("strategic_priorities", "Not specified")
        market_conditions = extra_context.get("market_conditions", "Not specified")
        
        # Budget recommendations (if available)
        recommended_allocation = extra_context.get("recommended_allocation", {})
        budget_summary = self._format_budget_recommendations(recommended_allocation)
        
        # Expected impact
        expected_impact = extra_context.get("expected_impact", {})
        impact_summary = self._format_expected_impact(expected_impact)
        
        # Organizational readiness
        org_readiness = extra_context.get("organizational_readiness", "Not assessed")
        
        summary = f"""
# EXECUTIVE SUMMARY - MMM OPTIMIZATION REVIEW

## Model Performance
- **Final MAPE**: {final_mape:.2f}% (lower is better, target <8% excellent, <10% acceptable)
- **Final R²**: {final_r2:.2f} (higher is better, target >0.85 strong, >0.75 acceptable)
- **Iterations**: {total_iterations} (preferred <15)
- **Trajectory**: {trajectory}
- **Convergence**: {"Converged" if total_iterations < 12 else "Extended optimization"}

## Business Context
- **Category**: {category}
- **Brand Positioning**: {brand_positioning.title()}
- **Market**: {market.upper()}
- **Strategic Priorities**: {strategic_priorities}
- **Market Conditions**: {market_conditions}

## Budget Reallocation Recommendations
{budget_summary}

## Expected Business Impact
{impact_summary}

## Organizational Readiness
{org_readiness}

## Optimization History
- Total Iterations: {total_iterations}
- Agent Cycle Pattern: {self._summarize_agent_cycles(history)}
- Major Milestones: {self._format_major_milestones(history)}
"""
        return summary
    
    def _format_budget_recommendations(self, allocation: Dict[str, Any]) -> str:
        """Format budget allocation recommendations for executive display."""
        if not allocation:
            return "  No specific allocation provided (using hyperparameter-implied effects)"
        
        lines = []
        for channel, data in allocation.items():
            current = data.get("current", 0)
            recommended = data.get("recommended", 0)
            change_pct = ((recommended - current) / current * 100) if current > 0 else 0
            
            change_indicator = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
            
            lines.append(
                f"  - **{channel.title()}**: ${current:,.0f} → ${recommended:,.0f} "
                f"({change_indicator} {abs(change_pct):.1f}%)"
            )
        
        return "\n".join(lines) if lines else "  Not specified"
    
    def _format_expected_impact(self, impact: Dict[str, Any]) -> str:
        """Format expected business impact for executive display."""
        if not impact:
            return "  Not provided (would come from model predictions)"
        
        sales_lift = impact.get("sales_lift_pct", "N/A")
        roi_improvement = impact.get("roi_improvement_pct", "N/A")
        efficiency_gain = impact.get("marketing_efficiency_gain_pct", "N/A")
        
        return f"""  - **Sales Lift**: {sales_lift}% (incremental vs current allocation)
  - **ROI Improvement**: {roi_improvement}% (average across channels)
  - **Marketing Efficiency Gain**: {efficiency_gain}% (cost per incremental sale)"""
    
    def _summarize_agent_cycles(self, history: MMMHistory) -> str:
        """Summarize agent interaction patterns."""
        if len(history) == 0:
            return "No history"
        
        # Count agent appearances
        agent_counts = {}
        for record in history.iterations:
            agent = record.agent_name
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Format as "DS:5, BA:3, RP:2"
        summary = ", ".join([f"{k.upper()[:2]}:{v}" for k, v in agent_counts.items()])
        return summary
    
    def _format_major_milestones(self, history: MMMHistory) -> str:
        """Extract and format major milestones from history."""
        if len(history) == 0:
            return "Initial run"
        
        milestones = []
        
        # Check for significant MAPE improvements
        if len(history) >= 2:
            first_mape = history.iterations[0].output.validation_results.get("mape") if history.iterations[0].output.validation_results else None
            last_mape = history.iterations[-1].output.validation_results.get("mape") if history.iterations[-1].output.validation_results else None
            
            if first_mape and last_mape:
                improvement = ((first_mape - last_mape) / first_mape * 100)
                milestones.append(f"MAPE improved {improvement:.1f}%")
        
        # Check for BA validation passes
        ba_validations = [r for r in history.iterations if r.agent_name == "business_analyst"]
        if ba_validations:
            last_ba = ba_validations[-1]
            if last_ba.output.validation_results:
                status = last_ba.output.validation_results.get("status", "UNKNOWN")
                milestones.append(f"BA: {status}")
        
        return ", ".join(milestones) if milestones else "In progress"
    
    def _extract_validation_status(self, history: MMMHistory) -> Dict[str, Any]:
        """
        Extract latest validation status from each agent type.
        
        Args:
            history: Optimization history
        
        Returns:
            Dictionary with validation status for DS, BA, RP
        """
        status = {
            "data_scientist": {"status": "NOT_RUN", "issues": []},
            "business_analyst": {"status": "NOT_RUN", "issues": []}
        }
        
        # Get most recent output from each agent
        for record in reversed(history.iterations):
            agent = record.agent_name
            
            if agent == "data_scientist" and status["data_scientist"]["status"] == "NOT_RUN":
                if record.output.validation_results:
                    # DS validation passed if convergence achieved and no errors
                    converged = record.output.flags.get("converged", False)
                    status["data_scientist"]["status"] = "APPROVED" if converged else "IN_PROGRESS"
                    status["data_scientist"]["issues"] = record.output.flags.get("issues", [])
            
            elif agent == "business_analyst" and status["business_analyst"]["status"] == "NOT_RUN":
                if record.output.validation_results:
                    ba_status = record.output.validation_results.get("status", "UNKNOWN")
                    issues = record.output.validation_results.get("issues", [])
                    status["business_analyst"]["status"] = ba_status
                    status["business_analyst"]["issues"] = issues
        
        return status
    
    def _check_auto_reject_conditions(
        self,
        state: MMMState,
        history: MMMHistory,
        validation_status: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check for conditions that auto-reject without LLM review.
        
        These are hard constraints that violate fundamental requirements:
        - Methodology violations (zero tolerance)
        - Critical unresolved Business Analyst flags
        - Catastrophic model quality (MAPE >15%, R² <0.60)
        
        Args:
            state: Current MMM state
            history: Optimization history
            validation_status: Extracted validation status
        
        Returns:
            Rejection reason string if auto-reject triggered, None otherwise
        """
        # Check for critical Business Analyst flags
        ba_issues = validation_status["business_analyst"].get("issues", [])
        critical_ba_issues = [i for i in ba_issues if i.get("severity") == "critical"]
        if critical_ba_issues:
            issue_summary = "; ".join([i.get("reasoning", "") for i in critical_ba_issues[:3]])
            return (
                f"CRITICAL BUSINESS ISSUES: Business Analyst flagged {len(critical_ba_issues)} "
                f"critical unresolved issues: {issue_summary}. Must be resolved before approval consideration."
            )
        
        # Check for catastrophic model quality
        mape = state.performance.get("mape", 100)
        r2 = state.performance.get("r_squared", 0)
        
        if mape > 15.0:
            return (
                f"CATASTROPHIC MODEL QUALITY: MAPE of {mape:.1f}% far exceeds acceptable "
                f"threshold (target <10%, absolute limit <15%). Model is not production-ready."
            )
        
        if r2 < 0.60:
            return (
                f"POOR EXPLANATORY POWER: R² of {r2:.2f} is below minimum threshold (0.60). "
                f"Model explains too little variance in sales to be reliable for decision-making."
            )
        
        return None
    
    def _create_rejection_output(
        self,
        reason: str,
        state: MMMState,
        history: MMMHistory
    ) -> AgentOutput:
        """
        Create rejection output for auto-reject conditions.
        
        Args:
            reason: Rejection reason string
            state: Current MMM state
            history: Optimization history
        
        Returns:
            AgentOutput with REJECTED decision
        """
        # Determine which agent should address the issues
        next_agent = None
        if "METHODOLOGY VIOLATION" in reason:
            next_agent = "data_scientist"  # DS must fix methodology
        elif "BUSINESS ISSUES" in reason:
            next_agent = "business_analyst"  # BA must resolve
        elif "MODEL QUALITY" in reason or "EXPLANATORY POWER" in reason:
            next_agent = "data_scientist"  # DS must improve model
        
        return AgentOutput(
            agent_name=self.name,
            action="rejection",
            proposed_changes=None,
            validation_results=None,
            decision="REJECTED",
            reasoning=f"AUTO-REJECT: {reason}",
            flags={
                "decision_type": "REJECTED",
                "auto_reject": True,
                "reject_reason": reason,
                "required_improvements": [
                    "Address critical issues flagged in rejection reason",
                    "Re-run optimization with corrected configuration",
                    "Ensure all validations pass before resubmission"
                ],
                "model_quality": {
                    "mape": state.performance.get("mape"),
                    "r_squared": state.performance.get("r_squared"),
                    "iterations": len(history),
                }
            },
            confidence=1.0,  # High confidence in rejection (hard constraint violated)
            next_agent_suggestion=next_agent
        )
    
    def _call_llm_for_decision(
        self,
        exec_summary: str,
        validation_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call Claude LLM for CMO-level strategic decision making.
        
        Sends comprehensive executive summary to Claude, asking it to:
        1. Assess model quality and business readiness
        2. Evaluate strategic fit and risk
        3. Make approval decision (APPROVED, CONDITIONAL_APPROVAL, APPROVED_FOR_TESTING, REJECTED)
        4. Define implementation plan, success criteria, monitoring
        5. Set conditions if conditional approval
        
        Args:
            exec_summary: Rich executive summary
            validation_status: Agent validation status
        
        Returns:
            Dictionary containing:
                - decision: One of four decision types
                - reasoning: Business narrative
                - implementation_plan: Timeline, phasing, geography
                - success_criteria: Measurable targets
                - risk_mitigation: Monitoring and rollback triggers
                - conditions: List of requirements (if conditional)
                - next_steps: Required work (if rejected)
        """
        system_prompt = """You are a Chief Marketing Officer (CMO) of a major FMCG/CPG company with 20+ years of experience leading marketing strategy and budget allocation.

Your expertise includes:
- Marketing Mix Modeling (MMM) interpretation and deployment
- Media channel effectiveness (TV: brand building, long carryover; Digital: performance, short-term; Print: declining; OOH: awareness)
- Brand equity management and long-term value protection
- Budget optimization balancing efficiency with brand health
- Risk assessment (execution, brand, financial, measurement)
- Organizational change management
- Stakeholder management (CEO, CFO, Board expectations)

Your role is to:
1. Review MMM optimization results as an executive summary
2. Assess model quality, business readiness, and strategic fit
3. Make final deployment decision with clear business reasoning
4. Define implementation plan, success criteria, and risk mitigation
5. Balance analytical rigor with business pragmatism

DECISION CRITERIA:
- **Model Quality**: MAPE <8% excellent, <10% acceptable; R² >0.85 strong, >0.75 acceptable
- **Validation Status**: All agents (DS, BA, RP) must have clean validations
- **Strategic Fit**: Recommendations align with brand positioning and priorities
- **Change Magnitude**: Large shifts (>20%) require phasing and monitoring
- **Brand Equity**: Protect TV/brand-building for premium brands
- **Execution Risk**: Organizational capability to implement changes

DECISION OPTIONS:
1. **APPROVED**: Full immediate national rollout
   - Excellent quality, all validations passed, moderate changes, low risk
   
2. **CONDITIONAL_APPROVAL**: Approved with conditions
   - Good quality but significant changes or risks
   - Require phased rollout, enhanced monitoring, or specific guardrails
   
3. **APPROVED_FOR_TESTING**: Geo pilot before national scale
   - Promising but unproven, want real-world validation
   - Test in 2-3 markets for 8-12 weeks, then decide on scale
   
4. **REJECTED**: Not ready for deployment
   - Critical unresolved issues, poor quality, strategic misalignment
   - Specify what needs improvement and who should address it

BRAND EQUITY CONSIDERATIONS (CRITICAL):
- **Premium Brands**: TV cuts >15% are risky, phase over 2+ quarters
- **Growth Brands**: Can accept higher digital bets if ROI is proven
- **Mature Brands**: Balance efficiency with awareness maintenance
- **TV/Sponsorships**: Not just immediate sales, builds long-term equity

Output strict JSON format:
{
  "decision": "APPROVED" | "CONDITIONAL_APPROVAL" | "APPROVED_FOR_TESTING" | "REJECTED",
  "reasoning": "business narrative (2-3 sentences, no jargon)",
  "implementation_plan": {
    "timeline": "Q1 2026, Q1-Q2 2026, Immediate, etc.",
    "phasing": "how changes roll out over time",
    "geography": "National, Regional test, Geo pilot, etc.",
    "key_actions": ["action 1", "action 2"]
  },
  "success_criteria": [
    "Sales lift ≥X%",
    "ROI improvement ≥Y%",
    "Brand awareness ≥Z%"
  ],
  "risk_mitigation": {
    "monitoring_frequency": "Daily, Weekly, Monthly",
    "key_metrics": ["metric 1", "metric 2"],
    "red_flags": ["trigger 1 for intervention"],
    "rollback_plan": "what to do if things go wrong"
  },
  "conditions": ["condition 1", "condition 2"],  // Only if CONDITIONAL_APPROVAL
  "next_steps": ["step 1", "step 2"]  // Only if REJECTED
}"""

        user_prompt = f"""{exec_summary}

## Agent Validation Status
- **Data Scientist**: {validation_status["data_scientist"]["status"]} 
  {f"(Issues: {len(validation_status['data_scientist']['issues'])})" if validation_status['data_scientist']['issues'] else "(Clean)"}
- **Business Analyst**: {validation_status["business_analyst"]["status"]}
  {f"(Issues: {len(validation_status['business_analyst']['issues'])})" if validation_status['business_analyst']['issues'] else "(Clean)"}

---

As CMO, review this executive summary and make your deployment decision. Consider:
1. Is the model quality sufficient for production deployment?
2. Do the validations give confidence in the recommendations?
3. Are the budget shifts strategically sound and implementable?
4. What are the risks (brand, execution, financial) and how to mitigate?
5. Should we proceed, test first, phase the rollout, or reject?

Return your decision in strict JSON format as specified."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            temperature=0.4,  # Moderate temperature for strategic reasoning
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse JSON response
        response_text = response.content[0].text.strip()
        
        # Handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(response_text)
    
    def _fallback_decision(
        self,
        state: MMMState,
        history: MMMHistory,
        validation_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback decision using rule-based logic when LLM fails.
        
        Applies conservative decision logic:
        - Check model quality thresholds
        - Check validation status
        - Apply safe defaults
        
        Args:
            state: Current MMM state
            history: Optimization history
            validation_status: Agent validation status
        
        Returns:
            Decision result in same format as LLM output
        """
        mape = state.performance.get("mape", 100)
        r2 = state.performance.get("r_squared", 0)
        iterations = len(history)
        
        # Check validations
        ba_status = validation_status["business_analyst"]["status"]
        ba_issues = validation_status["business_analyst"]["issues"]
        
        # Decision logic
        if mape <= self.MAPE_EXCELLENT and r2 >= self.R2_STRONG and not ba_issues:
            decision = "APPROVED"
            reasoning = (
                f"Excellent model quality (MAPE {mape:.1f}%, R² {r2:.2f}). "
                f"All validations passed. Ready for deployment."
            )
            conditions = []
            next_steps = []
        elif mape <= self.MAPE_GOOD and r2 >= self.R2_ACCEPTABLE:
            decision = "CONDITIONAL_APPROVAL"
            reasoning = (
                f"Good model quality (MAPE {mape:.1f}%, R² {r2:.2f}). "
                f"Conditionally approved with phased rollout to manage risk."
            )
            conditions = [
                "Phased implementation over 2 quarters",
                "Weekly monitoring for first month",
                "Monthly brand tracking"
            ]
            next_steps = []
        else:
            decision = "REJECTED"
            reasoning = (
                f"Model quality insufficient (MAPE {mape:.1f}%, R² {r2:.2f}). "
                f"Requires additional optimization."
            )
            conditions = []
            next_steps = [
                f"Improve MAPE to <{self.MAPE_GOOD}%",
                "Resolve business analyst issues",
                "Re-run optimization"
            ]
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "implementation_plan": {
                "timeline": "Q1-Q2 2026" if decision == "CONDITIONAL_APPROVAL" else "TBD",
                "phasing": "Phased rollout" if decision == "CONDITIONAL_APPROVAL" else "N/A",
                "geography": "National",
                "key_actions": ["Deploy model", "Monitor performance"]
            },
            "success_criteria": [
                "Sales lift ≥2.0%",
                f"MAPE remains <{mape + 1:.1f}%",
                "Brand awareness maintained"
            ],
            "risk_mitigation": {
                "monitoring_frequency": "Weekly",
                "key_metrics": ["MAPE", "Sales", "Brand awareness"],
                "red_flags": ["MAPE increases >2pts", "Sales decline >5%"],
                "rollback_plan": "Revert to previous allocation"
            },
            "conditions": conditions,
            "next_steps": next_steps
        }
    
    def _calculate_decision_confidence(
        self,
        decision: str,
        validation_status: Dict[str, Any],
        state: MMMState,
        history: MMMHistory
    ) -> float:
        """
        Calculate confidence score for decision.
        
        Confidence based on:
        - Model quality (MAPE, R²)
        - Validation cleanliness
        - Decision type
        
        Args:
            decision: Decision type
            validation_status: Agent validation status
            state: Current MMM state
            history: Optimization history
        
        Returns:
            Confidence score 0.0-1.0
        """
        mape = state.performance.get("mape", 100)
        r2 = state.performance.get("r_squared", 0)
        
        # Base confidence by decision type
        confidence_map = {
            "APPROVED": 0.90,
            "CONDITIONAL_APPROVAL": 0.75,
            "APPROVED_FOR_TESTING": 0.70,
            "REJECTED": 0.85  # High confidence in rejection
        }
        base_confidence = confidence_map.get(decision, 0.70)
        
        # Adjust for model quality
        if mape <= self.MAPE_EXCELLENT and r2 >= self.R2_STRONG:
            base_confidence += 0.05
        elif mape > self.MAPE_ACCEPTABLE or r2 < self.R2_ACCEPTABLE:
            base_confidence -= 0.10
        
        # Adjust for validation issues
        ba_issues = validation_status["business_analyst"]["issues"]
        if ba_issues:
            base_confidence -= (len(ba_issues) * 0.05)
        
        return max(0.5, min(1.0, base_confidence))
