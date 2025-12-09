"""
Main Orchestrator for MMM Multi-Agent Optimization

This is the central control system that coordinates all agents, manages
the optimization loop, and enforces the 5-phase workflow.

    Design: See Sections 3-7 of optimization.md for detailed architecture.

Key Responsibilities:
1. Initialize optimization session with baseline state
2. Coordinate agent selection via BERT router
3. Enforce methodology protection via LLM prompts (Data Scientist)
4. Track convergence and terminate session appropriately
5. Maintain complete audit trail in MMMHistory

3-Agent Workflow:
1. Technical Optimization (Data Scientist) - LLM-enforced methodology protection
2. Business Validation (Business Analyst) - ROI validation and benchmarking
3. Executive Approval (Stakeholder) - Final decision and deployment
"""

from typing import Dict, Any, Optional, List
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory, IterationRecord
from mmm_optimizer.a2a.client import A2AClient
from mmm_optimizer.routing.dynamic_bert_router import DynamicBERTRouter
from mmm_optimizer.agents.base_agent import AgentOutput


class Orchestrator:
    """
    Main orchestration engine for multi-agent MMM optimization.
    
    This class implements the complete optimization workflow, coordinating
    agents, enforcing methodology protection, and managing convergence.
    
    Architecture:
    - Uses BERTAgentRouter for intelligent agent selection
    - Uses A2AClient for uniform agent communication
    - Maintains MMMHistory for complete audit trail
    - Methodology protection via LLM prompts (Data Scientist agent)
    
    Workflow States:
    - INITIALIZING: Loading baseline model
    - TECHNICAL_OPT: Data Scientist tuning hyperparameters (methodology protected)
    - BUSINESS_VAL: Business Analyst ROI validation
    - EXECUTIVE_APPROVAL: Stakeholder final decision
    - CONVERGED: Optimization complete, approved for deployment
    - TERMINATED: Session ended (loop detection or max iterations)
    
    Example Usage:
        >>> orchestrator = Orchestrator(
        ...     baseline_state=initial_state,
        ...     max_iterations=20,
        ...     convergence_threshold=0.95
        ... )
        >>> 
        >>> results = orchestrator.run_session()
        >>> print(f"Final MAPE: {results['final_state'].performance.mape:.2f}%")
        >>> print(f"Deployment decision: {results['stakeholder_decision']}")
    """
    
    def __init__(
        self,
        baseline_state: MMMState,
        max_iterations: int = 20,
        convergence_threshold: float = 0.95,
        bert_model_path: Optional[str] = None,
        update_interval: int = 1,
        batch_update_size: int = 5,
        learning_rate: float = 1e-5,
        save_after_update: bool = True
    ):
        """
        Initialize orchestrator with baseline configuration.
        
        Args:
            baseline_state: Starting MMMState with initial model
            max_iterations: Maximum optimization iterations before forcing termination
            convergence_threshold: Convergence score threshold [0, 1] for completion
            bert_model_path: Path to trained BERT router checkpoint (default: ./mmm_optimizer/bert_router_trained)
            update_interval: Update model every N iterations (default: 1 for true online learning)
            batch_update_size: Minimum examples to accumulate before updating (default: 5)
            learning_rate: Learning rate for online updates (default: 1e-5)
            save_after_update: Whether to save model after each update (default: True)
        """
        self.baseline_state = baseline_state
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize components
        from datetime import datetime
        session_id = f"MMM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history = MMMHistory(session_id=session_id)
        self.a2a_client = A2AClient()
        
        # Initialize dynamic BERT router with online learning
        model_path = bert_model_path or "./mmm_optimizer/bert_router_trained"
        self.router = DynamicBERTRouter(
            pretrained_model_path=model_path,
            model_save_path=model_path,  # Save updates to same location
            update_interval=update_interval,
            batch_update_size=batch_update_size,
            learning_rate=learning_rate,
            save_after_update=save_after_update
        )
        print(f"✓ Dynamic BERT Router initialized with online learning")
        print(f"  - Update interval: every {update_interval} iteration(s)")
        print(f"  - Batch size: {batch_update_size} examples")
        print(f"  - Model saves to: {model_path}")
        
        # Session state
        self.current_state = baseline_state
        self.session_status = "INITIALIZING"
        self.last_agent_output: Optional[AgentOutput] = None
    
    def run_session(self) -> Dict[str, Any]:
        """
        Execute complete optimization session.
        
        This is the main entry point that runs the full optimization loop
        until convergence or termination.
        
        Workflow:
        1. Initialize session with baseline state
        2. Loop until convergence or max_iterations:
            a. Detect BA-DS loops (force Stakeholder escalation)
            b. Select next agent via BERT router
            c. Call agent via A2A protocol (methodology protected at DS LLM level)
            d. Update state and history
            e. Check convergence
        3. If converged: Call Stakeholder for final approval
        4. Return results with complete history
        
        Returns:
            Dict containing:
            - final_state: Final MMMState
            - history: Complete MMMHistory
            - stakeholder_decision: Final approval decision
            - session_status: Final status (CONVERGED/TERMINATED)
            - iterations_completed: Total iteration count
            - convergence_score: Final convergence score
            - termination_reason: Why session ended
        
        Example:
            >>> results = orchestrator.run_session()
            >>> if results["stakeholder_decision"] == "APPROVED":
            ...     deploy_model(results["final_state"])
        """
        self.session_status = "TECHNICAL_OPT"
        iteration = 1
        termination_reason = None
        
        print(f"Starting optimization session...")
        print(f"Baseline MAPE: {self.baseline_state.performance.mape:.2f}%")
        print(f"Max iterations: {self.max_iterations}")
        print()
        
        while iteration <= self.max_iterations:
            print(f"=== Iteration {iteration} ===")
            
            # Store state before agent execution (for dynamic learning)
            state_before_agent = self.current_state
            
            # LOOP BREAKER: Check if stuck in BA-DS cycle
            if self.history.is_stuck_in_loop(threshold=3):
                print("⚠️  LOOP DETECTED: BA-DS cycle detected. Forcing escalation to Stakeholder.")
                selected_agent = "stakeholder"
                confidence = 1.0
            else:
                # Phase 1: Select next agent via dynamic BERT routing
                selected_agent, confidence = self.router.route(
                    self.current_state,
                    self.history,
                    return_confidence=True
                )
            print(f"BERT Router → {selected_agent} (confidence: {confidence:.2f})")
            
            # Phase 2: Call selected agent
            state_before = self.current_state.copy()  # Capture state before changes
            agent_output = self._call_agent(selected_agent, iteration)
            print(f"{agent_output.agent_name} → {agent_output.decision}")
            print(f"Reasoning: {agent_output.reasoning[:100]}...")
            
            # Phase 3: Apply agent output and update state
            if agent_output.decision not in ["REJECTED", "NO_CHANGE"]:
                self.current_state = self._apply_agent_changes(
                    self.current_state,
                    agent_output
                )
            
            # Phase 4: Record iteration with proper state tracking
            record = IterationRecord(
                iteration=iteration,
                agent_name=agent_output.agent_name,
                action=agent_output.action,
                state_before=state_before,
                state_after=self.current_state,
                outcome=self._determine_outcome(agent_output, state_before, self.current_state),
                agent_output_raw=agent_output.reasoning,
                agent_output_structured=agent_output
            )
            self.history.add_record(record)
            
            # Phase 4.5: Record iteration for online learning
            outcome = {
                "mape": self.current_state.performance.mape,
                "r2": self.current_state.performance.r2,
                "decision": agent_output.decision,
                "action": agent_output.action
            }
            self.router.record_iteration(
                state=state_before_agent,
                history=self.history,
                actual_agent_used=selected_agent,
                outcome=outcome
            )
            
            # Phase 5: Check convergence
            self.current_state.convergence_score = self._calculate_convergence(
                self.history
            )
            print(f"Convergence score: {self.current_state.convergence_score:.2f}")
            
            if self.current_state.convergence_score >= self.convergence_threshold:
                print("✓ Convergence threshold reached")
                
                # Call Business Analyst for final validation
                ba_output = self._call_agent("business_analyst", iteration)
                if ba_output.decision == "APPROVED":
                    print("✓ Business Analyst approved")
                    
                    # Call Stakeholder for executive approval
                    sh_output = self._call_agent("stakeholder", iteration)
                    self.session_status = "CONVERGED"
                    termination_reason = "Converged and approved by Stakeholder"
                    
                    return {
                        "final_state": self.current_state,
                        "history": self.history,
                        "stakeholder_decision": sh_output.decision,
                        "session_status": self.session_status,
                        "iterations_completed": iteration,
                        "convergence_score": self.current_state.convergence_score,
                        "termination_reason": termination_reason
                    }
            
            iteration += 1
            print()
        
        # Max iterations reached
        self.session_status = "TERMINATED"
        termination_reason = f"Max iterations ({self.max_iterations}) reached"
        print(f"⚠️  {termination_reason}")
        
        return {
            "final_state": self.current_state,
            "history": self.history,
            "stakeholder_decision": "TERMINATED",
            "session_status": self.session_status,
            "iterations_completed": iteration - 1,
            "convergence_score": self.current_state.convergence_score,
            "termination_reason": termination_reason
        }
    
    def _call_agent(self, agent_name: str, iteration: int) -> AgentOutput:
        """
        Call an agent via A2A protocol.
        
        Args:
            agent_name: Name of agent to call
            iteration: Current iteration number
        
        Returns:
            AgentOutput from the agent
        """
        context = {
            "state": self.current_state,
            "history": self.history,
            "extra_context": {
                "iteration": iteration,
                "last_agent_output": self.last_agent_output
            }
        }
        
        output = self.a2a_client.call_agent(
            agent_name=agent_name,
            task=f"optimization_iteration_{iteration}",
            context=context
        )
        
        self.last_agent_output = output
        return output
    
    def _apply_agent_changes(
        self,
        state: MMMState,
        agent_output: AgentOutput
    ) -> MMMState:
        """
        Apply agent's proposed changes to state.
        
        Args:
            state: Current state
            agent_output: Agent output with proposed_changes
        
        Returns:
            Updated MMMState with proposed changes applied
        """
        import copy
        
        # Create deep copy to avoid mutating current state
        new_state = state.copy()
        new_state.iteration = state.iteration + 1
        
        # Apply proposed changes if any exist
        if agent_output.proposed_changes:
            # Merge hyperparameter changes
            if "hyperparameters" in agent_output.proposed_changes:
                for param, value in agent_output.proposed_changes["hyperparameters"].items():
                    new_state.update_hyperparameter(param, value)
            
            # Update performance metrics if provided
            if "performance" in agent_output.proposed_changes:
                perf = agent_output.proposed_changes["performance"]
                new_state.update_performance(
                    mape=perf.get("mape"),
                    r2=perf.get("r2"),
                    mae=perf.get("mae")
                )
            
            # Update business context (flags, constraints)
            if "business_context" in agent_output.proposed_changes:
                biz_changes = agent_output.proposed_changes["business_context"]
                for key, value in biz_changes.items():
                    new_state.business_context[key] = value
        
        return new_state
    
    def _determine_outcome(
        self,
        agent_output: AgentOutput,
        state_before: MMMState,
        state_after: MMMState
    ) -> str:
        """
        Determine the outcome description for this iteration.
        
        Args:
            agent_output: Agent's output
            state_before: State before agent ran
            state_after: State after changes applied
        
        Returns:
            Outcome string: "improvement", "validated", "approved", "rejected", "no_change"
        """
        if agent_output.decision == "REJECTED":
            return "rejected"
        elif agent_output.decision == "NO_CHANGE":
            return "no_change"
        elif agent_output.agent_name == "stakeholder":
            return "approved" if agent_output.decision in ["APPROVED", "CONDITIONAL_APPROVAL"] else "rejected"
        elif agent_output.agent_name == "business_analyst":
            return "validated"
        elif agent_output.agent_name == "data_scientist":
            # Check if MAPE improved
            if state_after.performance.mape < state_before.performance.mape:
                return "improvement"
            else:
                return "no_improvement"
        else:
            return "executed"
    
    def _calculate_convergence(self, history: MMMHistory) -> float:
        """
        Calculate convergence score based on recent performance stability.
        
        Uses Coefficient of Variation (CV) to measure MAPE stability over
        the last N iterations. Lower CV indicates stable performance.
        
        Args:
            history: MMMHistory with past iterations
        
        Returns:
            Convergence score in [0, 1]
            - 1.0 = Fully converged (CV < 1%, MAPE stable and good)
            - 0.0 = Not converged (CV > 5% or still improving rapidly)
        
        Logic:
        - Look at last 5 iterations
        - Calculate coefficient of variation: CV = std(MAPE) / mean(MAPE)
        - CV < 1% indicates high stability → convergence = 1.0
        - CV > 5% indicates instability → convergence = 0.0
        - If MAPE still improving >2%, reduce convergence (keep optimizing)
        
        Example:
            - MAPE trajectory: [10.5, 10.3, 10.4, 10.2, 10.3]
            - Mean: 10.34, Std: 0.12
            - CV: 0.12/10.34 = 0.0116 (1.16%)
            - Convergence: 1.0 - min(0.0116/0.05, 1.0) = 0.768
        """
        import numpy as np
        
        # Need at least 5 iterations to assess stability
        if len(history) < 5:
            return 0.0
        
        # Extract MAPE values from last 5 iterations
        recent_iterations = history.iterations[-5:]
        mape_values = [
            record.state_after.performance.mape 
            for record in recent_iterations
            if record.state_after.performance.mape is not None
        ]
        
        # Need exactly 5 valid MAPE values
        if len(mape_values) < 5:
            return 0.0
        
        # Calculate coefficient of variation (CV)
        mean_mape = np.mean(mape_values)
        std_mape = np.std(mape_values)
        
        # Avoid division by zero AND unrealistic MAPE
        if mean_mape < 0.1:  # MAPE < 0.1% is unrealistic or invalid
            return 0.0
        
        cv = std_mape / mean_mape
        
        # Base convergence score: CV < 5% threshold
        # CV = 0% → score = 1.0
        # CV = 5% → score = 0.0
        # Linear interpolation between
        convergence_score = max(0.0, 1.0 - (cv / 0.05))
        
        # Additional check: If MAPE still improving rapidly, reduce convergence
        # (Even if stable, we want to keep optimizing if there's clear improvement)
        if len(mape_values) >= 5:
            improvement = mape_values[0] - mape_values[-1]  # First vs last in window
            improvement_rate = improvement / mape_values[0]
            
            # If improving >2% over 5 iterations, reduce convergence
            if improvement_rate > 0.02:
                convergence_score *= 0.7  # Keep optimizing
        
        # Optional: Penalize convergence if MAPE still high
        # (Don't converge at bad performance)
        if mean_mape > 12.0:  # Configurable threshold
            convergence_score *= 0.5
        
        return max(0.0, min(1.0, convergence_score))  # Clamp to [0, 1]
