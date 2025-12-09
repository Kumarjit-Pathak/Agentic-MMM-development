"""
State Management for MMM Optimization

Defines core data structures for tracking optimization state and history:
- MMMState: Current model configuration and performance
- IterationRecord: Single iteration details
- MMMHistory: Complete optimization trajectory

These structures are passed between the orchestrator and agents to maintain
context throughout the optimization process.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Performance:
    """
    Performance metrics for an MMM model.
    
    Attributes:
        mape: Mean Absolute Percentage Error (lower is better)
        r2: R-squared coefficient of determination (higher is better, 0-1)
        mae: Mean Absolute Error (lower is better)
    """
    mape: float
    r2: float
    mae: float


@dataclass
class MMMState:
    """
    Represents the current state of an MMM model during optimization.
    
    This includes performance metrics, model configuration (methodology + hyperparameters),
    and business context. The state is updated after each agent action.
    
    Attributes:
        iteration: Current iteration number (0-based)
        performance: Dict with metrics like {mape: 9.2, r2: 0.88, mae: 1245.3}
        model_config: Nested dict containing:
            - methodology: Protected fields (adstock_type, saturation_type, hierarchy_levels, etc.)
            - hyperparameters: Tunable params (tv_adstock_decay, learning_rate, etc.)
        business_context: Dict with:
            - flags: List of business validation issues
            - constraints: Business rules (roi_caps, min_spend, etc.)
            - brand: Brand identifier
            - category: Product category (CPG, Auto, etc.)
            - market: Geographic market
        convergence_score: Estimated convergence (0.0-1.0, higher = closer to optimal)
        timestamp: When this state was created
    
    Example:
        >>> state = MMMState(
        ...     iteration=5,
        ...     performance={"mape": 8.5, "r2": 0.89, "mae": 1103.2},
        ...     model_config={
        ...         "methodology": {
        ...             "adstock_type": "beta_gamma",
        ...             "saturation_type": "hill",
        ...             "hierarchy_levels": ["brand", "region", "channel"],
        ...             "architecture_type": "hierarchical_neural_additive",
        ...             "loss_type": "mse"
        ...         },
        ...         "hyperparameters": {
        ...             "tv_adstock_decay": 0.70,
        ...             "digital_adstock_decay": 0.55,
        ...             "tv_saturation_alpha": 2.5,
        ...             "learning_rate": 0.001,
        ...             "regularization": 0.01
        ...         }
        ...     },
        ...     business_context={
        ...         "flags": [],
        ...         "constraints": {"tv_roi_min": 2.5, "digital_roi_max": 6.0},
        ...         "brand": "Brand_A",
        ...         "category": "CPG",
        ...         "market": "US"
        ...     },
        ...     convergence_score=0.75
        ... )
    """
    
    iteration: int
    performance: Performance  # Performance metrics
    model_config: Dict[str, Dict[str, Any]]  # {methodology: {...}, hyperparameters: {...}}
    business_context: Dict[str, Any]  # {flags, constraints, brand, category, market}
    convergence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def copy(self) -> 'MMMState':
        """Create a deep copy of this state."""
        import copy
        return copy.deepcopy(self)
    
    def get_methodology(self) -> Dict[str, Any]:
        """Get the methodology configuration (protected fields)."""
        return self.model_config.get("methodology", {})
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get the hyperparameters (tunable fields)."""
        return self.model_config.get("hyperparameters", {})
    
    def update_hyperparameter(self, param_name: str, new_value: Any) -> None:
        """
        Update a single hyperparameter.
        
        Args:
            param_name: Name of the hyperparameter (e.g., "tv_adstock_decay")
            new_value: New value to set
        """
        if "hyperparameters" not in self.model_config:
            self.model_config["hyperparameters"] = {}
        self.model_config["hyperparameters"][param_name] = new_value
    
    def update_performance(self, mape: Optional[float] = None, r2: Optional[float] = None, mae: Optional[float] = None) -> None:
        """Update performance metrics."""
        if mape is not None:
            self.performance.mape = mape
        if r2 is not None:
            self.performance.r2 = r2
        if mae is not None:
            self.performance.mae = mae
    
    def has_business_flags(self) -> bool:
        """Check if there are any business validation flags."""
        return len(self.business_context.get("flags", [])) > 0


@dataclass
class IterationRecord:
    """
    Records details of a single optimization iteration.
    
    Each iteration involves calling one agent, which may propose changes,
    validate results, or make approval decisions. This record captures
    the complete state transition.
    
    Attributes:
        iteration: Iteration number
        agent_name: Which agent was called ("data_scientist", "business_analyst", etc.)
        action: Type of action taken ("hyperparameter_update", "business_validation", etc.)
        state_before: MMMState before agent ran
        state_after: MMMState after changes applied (may be same as before if rejected)
        outcome: Result description ("improvement", "validated", "approved", "rejected")
        agent_output_raw: Raw text output from agent
        agent_output_structured: Parsed AgentOutput object
        timestamp: When this iteration occurred
    
    Example:
        >>> record = IterationRecord(
        ...     iteration=3,
        ...     agent_name="data_scientist",
        ...     action="hyperparameter_update",
        ...     state_before=state_before,
        ...     state_after=state_after,
        ...     outcome="improvement",
        ...     agent_output_raw="Adjusted tv_adstock_decay from 0.7 to 0.75",
        ...     agent_output_structured=agent_output_obj
        ... )
    """
    
    iteration: int
    agent_name: str
    action: str
    state_before: MMMState
    state_after: MMMState
    outcome: str
    agent_output_raw: str
    agent_output_structured: Any  # AgentOutput object
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_mape_change(self) -> Optional[float]:
        """Calculate MAPE change (negative = improvement)."""
        before = self.state_before.performance.mape
        after = self.state_after.performance.mape
        if before is not None and after is not None:
            return after - before
        return None
    
    def was_improvement(self) -> bool:
        """Check if this iteration improved MAPE."""
        change = self.get_mape_change()
        return change is not None and change < 0


@dataclass
class MMMHistory:
    """
    Tracks complete optimization history across all iterations.
    
    This is used by:
    - BERT Router: To understand patterns and select next agent
    - Agents: To see what's already been tried and avoid repetition
    - Loop Detection: To identify BA-DS cycles and force escalation
    - Final reporting: To summarize the optimization session
    
    Attributes:
        session_id: Unique identifier for this optimization session
        iterations: List of IterationRecord objects (chronological order)
        start_time: When optimization started
        end_time: When optimization completed (or None if running)
        final_decision: Final approval status (or None if not yet decided)
    
    Example:
        >>> history = MMMHistory(session_id="Brand_A_Q4_2025")
        >>> history.add_iteration(iteration_record_1)
        >>> history.add_iteration(iteration_record_2)
        >>> print(f"Total iterations: {len(history)}")
        >>> print(f"Last agent: {history.get_last_agent()}")
    """
    
    session_id: str
    iterations: List[IterationRecord] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    final_decision: Optional[str] = None
    
    def add_iteration(self, record: IterationRecord) -> None:
        """Add a new iteration record."""
        self.iterations.append(record)
    
    def add_record(self, record: IterationRecord) -> None:
        """Alias for add_iteration for backward compatibility."""
        self.add_iteration(record)
    
    def __len__(self) -> int:
        """Return number of iterations."""
        return len(self.iterations)
    
    def get_last_iteration(self) -> Optional[IterationRecord]:
        """Get the most recent iteration."""
        return self.iterations[-1] if self.iterations else None
    
    def get_last_agent(self) -> Optional[str]:
        """Get the name of the last agent called."""
        last = self.get_last_iteration()
        return last.agent_name if last else None
    
    def get_last_n_agents(self, n: int) -> List[str]:
        """
        Get the last N agents called (most recent first).
        
        Example:
            >>> history.get_last_n_agents(3)
            ['data_scientist', 'data_scientist', 'business_analyst']
        """
        return [rec.agent_name for rec in self.iterations[-n:]]
    
    def get_mape_trend(self, last_n: int = 3) -> List[float]:
        """
        Get MAPE values for the last N iterations.
        
        Used by BERT router to detect diminishing returns pattern.
        """
        trend = []
        for rec in self.iterations[-last_n:]:
            mape = rec.state_after.performance.mape
            if mape is not None:
                trend.append(mape)
        return trend
    
    def get_agent_cycle_count(self, agent_sequence: List[str], lookback: int = 10) -> int:
        """
        Count how many times a specific agent sequence has repeated recently.
        
        This detects ping-pong patterns like [BA, DS, BA, DS, BA, DS].
        
        Args:
            agent_sequence: List of agent names to check (e.g., ["business_analyst", "data_scientist"])
            lookback: How many recent iterations to check
        
        Returns:
            Number of times the sequence appears consecutively
        
        Example:
            >>> # If last 6 agents were: BA, DS, BA, DS, BA, DS
            >>> history.get_agent_cycle_count(["business_analyst", "data_scientist"], 10)
            3  # The pattern repeated 3 times
        """
        if not agent_sequence or len(agent_sequence) == 0:
            return 0
        
        recent_agents = self.get_last_n_agents(lookback)
        
        # Count consecutive occurrences of the pattern
        pattern_len = len(agent_sequence)
        cycle_count = 0
        
        for i in range(0, len(recent_agents) - pattern_len + 1, pattern_len):
            chunk = recent_agents[i:i + pattern_len]
            if chunk == agent_sequence:
                cycle_count += 1
            else:
                # Pattern broken, stop counting
                break
        
        return cycle_count
    
    def is_stuck_in_loop(self, threshold: int = 3) -> bool:
        """
        Detect if optimization is stuck in BA-DS or DS-BA loop.
        
        Args:
            threshold: How many cycles before considering it a loop (default 3)
        
        Returns:
            True if stuck in a loop
        """
        ba_ds_cycles = self.get_agent_cycle_count(["business_analyst", "data_scientist"], lookback=12)
        ds_ba_cycles = self.get_agent_cycle_count(["data_scientist", "business_analyst"], lookback=12)
        
        return ba_ds_cycles >= threshold or ds_ba_cycles >= threshold
    
    def get_mape_improvement_rate(self, last_n: int = 3) -> float:
        """
        Calculate average MAPE improvement per iteration.
        
        Used to detect diminishing returns (when improvements get smaller).
        
        Args:
            last_n: Number of recent iterations to analyze
        
        Returns:
            Average percentage improvement per iteration (negative = getting worse)
        
        Example:
            >>> # MAPE went 10.0 → 9.5 → 9.3 → 9.2
            >>> history.get_mape_improvement_rate(3)
            0.267  # Average 0.267% improvement per iteration
        """
        mape_values = self.get_mape_trend(last_n + 1)  # Need N+1 values for N improvements
        
        if len(mape_values) < 2:
            return 0.0
        
        improvements = [mape_values[i] - mape_values[i+1] for i in range(len(mape_values)-1)]
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def format_for_bert(self, current_state: MMMState) -> str:
        """
        Format history as text for BERT routing input.
        
        This creates a narrative description of the optimization trajectory
        that BERT can analyze to predict the optimal next agent.
        
        Format matches design doc Section 4.2:
        "[CLS] HISTORY: Iteration 1: Agent=data_scientist, MAPE=12.3→11.1, 
        Result=improvement. Iteration 2: ... CURRENT: MAPE=9.5, R2=0.88, ... [SEP]"
        
        Args:
            current_state: Current MMMState to include in summary
            
        Returns:
            Formatted text string for BERT input
        """
        text = "[CLS] HISTORY: "
        
        for rec in self.iterations:
            mape_before = rec.state_before.performance.mape
            mape_after = rec.state_after.performance.mape
            
            text += f"Iteration {rec.iteration}: "
            text += f"Agent={rec.agent_name}, "
            text += f"MAPE={mape_before:.1f}→{mape_after:.1f}, "
            text += f"Result={rec.outcome}. "
        
        # Add current state summary
        text += f"CURRENT: "
        text += f"MAPE={current_state.performance.mape:.1f}%, "
        text += f"R2={current_state.performance.r2:.2f}, "
        text += f"Flags={len(current_state.business_context.get('flags', []))}, "
        
        # Add recent agent pattern
        last_agents = self.get_last_n_agents(3)
        text += f"LastAgents={last_agents}. "
        
        text += "TASK: Select next agent [SEP]"
        
        return text
    
    def to_summary(self) -> Dict[str, Any]:
        """
        Create a summary dict for final reporting.
        
        Returns:
            Dict with session statistics and key events
        """
        return {
            "session_id": self.session_id,
            "total_iterations": len(self),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "final_decision": self.final_decision,
            "agent_calls": {
                agent: len([r for r in self.iterations if r.agent_name == agent])
                for agent in ["data_scientist", "business_analyst", "stakeholder"]
            },
            "mape_trajectory": self.get_mape_trend(len(self)),
        }
