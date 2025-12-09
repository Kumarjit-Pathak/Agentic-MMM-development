"""
Base Agent Classes for MMM Optimization

Defines the abstract BaseAgent class and AgentOutput dataclass that all
specialized agents must implement. This provides a consistent interface
for the orchestrator to interact with any agent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from mmm_optimizer.orchestrator.state import MMMState, MMMHistory


@dataclass
class AgentOutput:
    """
    Standardized output structure from any agent.
    
    All agents return this structure, allowing the orchestrator to handle
    outputs uniformly regardless of which agent was called.
    
    Attributes:
        agent_name: Name of the agent that produced this output
        action: Type of action taken (see Action Types below)
        proposed_changes: Dict of changes to apply (or None if validation/approval only)
        validation_results: Dict with validation details (or None if not applicable)
        decision: Approval decision (or None if not an approval action)
        reasoning: Narrative explanation of the agent's logic
        flags: Dict of issues, warnings, or recommendations
        confidence: Agent's confidence in its recommendation (0.0-1.0)
        next_agent_suggestion: Optional hint about which agent should go next
    
    Action Types:
        - "hyperparameter_update": Data Scientist tuning parameters
        - "business_validation": Business Analyst checking outputs
        - "full_approval": Stakeholder fully approving model
        - "conditional_approval": Stakeholder approving with conditions
        - "test_approval": Stakeholder approving for testing only
        - "rejection": Stakeholder rejecting proposal
    
    Decision Values (for Stakeholder):
        - "APPROVED": Full approval, ready for deployment
        - "CONDITIONAL_APPROVAL": Approved with conditions
        - "APPROVED_FOR_TESTING": Approved for pilot/test only
        - "REJECTED": Needs more work
    
    Example (Data Scientist):
        >>> AgentOutput(
        ...     agent_name="data_scientist",
        ...     action="hyperparameter_update",
        ...     proposed_changes={
        ...         "hyperparameters": {
        ...             "tv_adstock_decay": 0.70,
        ...             "tv_saturation_alpha": 2.5
        ...         }
        ...     },
        ...     validation_results=None,
        ...     decision=None,
        ...     reasoning="Extended TV carryover to ~10 weeks (industry standard for CPG)",
        ...     flags={"needs_business_review": True},
        ...     confidence=0.85,
        ...     next_agent_suggestion="business_analyst"
        ... )
    
    Example (Business Analyst):
        >>> AgentOutput(
        ...     agent_name="business_analyst",
        ...     action="business_validation",
        ...     proposed_changes=None,
        ...     validation_results={
        ...         "status": "PARTIAL_ALIGNMENT",
        ...         "issues": [
        ...             {"field": "digital_roi", "value": 8.5, "benchmark": "3-6x",
        ...              "severity": "warning"}
        ...         ]
        ...     },
        ...     decision=None,
        ...     reasoning="Digital ROI exceeds typical benchmarks",
        ...     flags={"investigate_digital_attribution": True},
        ...     confidence=0.75,
        ...     next_agent_suggestion="data_scientist"
        ... )
    
    Example (Stakeholder):
        >>> AgentOutput(
        ...     agent_name="stakeholder",
        ...     action="conditional_approval",
        ...     proposed_changes=None,
        ...     validation_results=None,
        ...     decision="CONDITIONAL_APPROVAL",
        ...     reasoning="Model quality acceptable, but need phased implementation",
        ...     flags={
        ...         "conditions": [
        ...             "Phased TV reduction: -10% Q1, -10% Q2",
        ...             "Monthly brand tracking study"
        ...         ],
        ...         "timeline": "Q1-Q2 2026",
        ...         "success_criteria": "Sales lift >2.5%, Brand awareness >85%"
        ...     },
        ...     confidence=0.90,
        ...     next_agent_suggestion=None
        ... )
    
    """
    
    agent_name: str
    action: str
    reasoning: str
    proposed_changes: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    next_agent_suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "reasoning": self.reasoning,
            "proposed_changes": self.proposed_changes,
            "validation_results": self.validation_results,
            "decision": self.decision,
            "flags": self.flags,
            "confidence": self.confidence,
            "next_agent_suggestion": self.next_agent_suggestion,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all MMM optimization agents.
    
    All specialized agents (Data Scientist, Business Analyst, Stakeholder)
    inherit from this class and implement the run() method.
    
    The orchestrator calls agents through this uniform interface, enabling
    easy addition of new agent types in the future.
    
    Attributes:
        name: Agent identifier (e.g., "data_scientist", "business_analyst")
        description: Human-readable description of agent's purpose
        
    Agent Roles & Restrictions:
        - **Data Scientist**: Tunes hyperparameters only. LLM-enforced methodology
          protection prevents changing adstock type, saturation type, hierarchy,
          architecture, or loss function.
        - **Business Analyst**: Validates outputs against industry benchmarks,
          flags issues, proposes business constraints. Cannot change model.
        - **Stakeholder**: Makes final approval/rejection decisions with business
          reasoning. Cannot modify model or override technical validations.
        
    Agent Contract:
        - Agents MUST NOT modify state directly (return proposed changes instead)
        - Agents SHOULD set appropriate confidence scores (0.0-1.0)
        - Agents MAY suggest next agent via next_agent_suggestion
        - All methodology protection is enforced at LLM prompt level (Data Scientist)
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier
            description: Purpose description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(
        self,
        state: MMMState,
        history: MMMHistory,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Execute agent logic and return proposed actions.
        
        This is the main entry point called by the orchestrator. The agent
        analyzes the current state and history, then returns recommendations.
        
        Args:
            state: Current MMMState (iteration, performance, config, context)
            history: Complete MMMHistory of all previous iterations
            extra_context: Optional additional context (e.g., business benchmarks,
                          strategic priorities, market conditions)
        
        Returns:
            AgentOutput with proposed changes, validation results, or decision
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If agent encounters unexpected error
        
        Implementation Notes:
            - Use state.copy() if you need to simulate changes
            - Check history.get_rejection_count(self.name) if relevant
            - Use history.format_for_bert() to see narrative trajectory
            - Set confidence based on how certain you are
            - In extra_context, look for:
                - "business_benchmarks": Industry benchmarks for validation (BA)
                - "strategic_priorities": Business strategy context (Stakeholder)
                - "market_conditions": External factors (Stakeholder)
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
