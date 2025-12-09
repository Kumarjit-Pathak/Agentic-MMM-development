"""
MMM Multi-Agent Optimization System

This package implements an intelligent multi-agent system for automating
Marketing Mix Model (MMM) optimization. The system replaces manual, weeks-long
hyperparameter tuning with an automated process that takes 2-3 days.

Core Components:
- Orchestrator: Manages workflow, state, and iteration history
- BERT Router: Learns optimal agent routing from historical sessions
- 4 Specialized Agents: Data Scientist, Business Analyst, Stakeholder, Research Persona
- Methodology Guard: Ensures zero methodology violations
- A2A Protocol: Standardized agent-to-agent communication

See optimization.md for complete technical documentation.
"""

__version__ = "1.0.0"
__author__ = "MMM Development Team"

from mmm_optimizer.orchestrator.state import MMMState, IterationRecord, MMMHistory, Performance
from mmm_optimizer.agents.base_agent import BaseAgent, AgentOutput

__all__ = [
    "MMMState",
    "IterationRecord", 
    "MMMHistory",
    "Performance",
    "BaseAgent",
    "AgentOutput",
]
