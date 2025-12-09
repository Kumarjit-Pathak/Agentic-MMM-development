"""
Entity and Intent Classifier

Optional component for extracting structured information from agent outputs
and user inputs. Useful for parsing natural language into actionable parameters.

Design: See Section 10 of optimization.md for entity/intent classification.

This is a LOWER-PRIORITY component and can be implemented later.
Initial version uses simple keyword matching.
"""

from typing import Dict, List, Any, Optional


class EntityIntentClassifier:
    """
    Extract entities and intents from natural language text.
    
    This classifier is useful for:
    1. Parsing agent reasoning into structured changes
    2. Extracting hyperparameter names/values from text
    3. Identifying user intent in interactive sessions
    
    Example Use Cases:
    - Agent says: "Increase tv_adstock_decay from 0.6 to 0.7"
      → Extract: entity="tv_adstock_decay", old_value=0.6, new_value=0.7
    
    - User says: "Focus on optimizing digital channels"
      → Extract: intent="optimize", entity_type="channel", entity="digital"
    
    Implementation Options:
    1. Simple (current): Regex and keyword matching
    2. Medium: spaCy NER with custom training
    3. Advanced: Fine-tuned BERT for entity extraction
    
    Example Usage:
        >>> classifier = EntityIntentClassifier()
        >>> entities = classifier.extract_entities(
        ...     "Increase tv_decay to 0.8 and digital_alpha to 2.5"
        ... )
        >>> print(entities)
        [
            {"param": "tv_decay", "value": 0.8, "action": "increase"},
            {"param": "digital_alpha", "value": 2.5, "action": "increase"}
        ]
    """
    
    def __init__(self):
        """Initialize classifier with keyword patterns."""
        # Action keywords
        self.action_keywords = {
            "increase": ["increase", "raise", "boost", "higher"],
            "decrease": ["decrease", "reduce", "lower", "drop"],
            "set": ["set", "change", "update", "modify"],
            "remove": ["remove", "delete", "drop"]
        }
        
        # Known MMM parameter patterns
        self.parameter_patterns = [
            "adstock_decay", "saturation_alpha", "learning_rate",
            "tv_decay", "digital_decay", "print_decay",
            "carryover", "roi", "spend"
        ]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract parameter entities from text.
        
        Args:
            text: Natural language text (e.g., agent reasoning)
        
        Returns:
            List of dicts with keys:
            - param: Parameter name
            - value: Numeric value (if found)
            - action: Action type ("increase", "decrease", "set")
            - confidence: Float in [0, 1]
        
        Example:
            >>> classifier.extract_entities("Set tv_decay to 0.75")
            [{"param": "tv_decay", "value": 0.75, "action": "set", "confidence": 0.8}]
        """
        # TODO: Implement entity extraction
        # This is a stub - implement with regex or spaCy
        
        entities = []
        text_lower = text.lower()
        
        # Simple keyword matching stub
        for param in self.parameter_patterns:
            if param in text_lower:
                entities.append({
                    "param": param,
                    "value": None,  # TODO: Extract numeric value
                    "action": "modify",
                    "confidence": 0.5
                })
        
        return entities
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify user/agent intent from text.
        
        Args:
            text: Natural language text
        
        Returns:
            Dict with:
            - intent: One of ["optimize", "validate", "approve", "reject", "query"]
            - confidence: Float in [0, 1]
            - entities: List of related entities
        
        Example:
            >>> classifier.classify_intent("Optimize TV campaign ROI")
            {"intent": "optimize", "confidence": 0.7, "entities": ["tv", "roi"]}
        """
        # TODO: Implement intent classification
        # This is a stub
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["optimize", "improve", "tune"]):
            return {"intent": "optimize", "confidence": 0.6, "entities": []}
        elif any(word in text_lower for word in ["validate", "check", "verify"]):
            return {"intent": "validate", "confidence": 0.6, "entities": []}
        elif any(word in text_lower for word in ["approve", "accept", "proceed"]):
            return {"intent": "approve", "confidence": 0.6, "entities": []}
        elif any(word in text_lower for word in ["reject", "decline", "stop"]):
            return {"intent": "reject", "confidence": 0.6, "entities": []}
        else:
            return {"intent": "query", "confidence": 0.4, "entities": []}
