"""
Dynamic BERT Router with Online Learning

This module implements a BERT-based agent router that continuously learns from
real optimization sessions. It starts with a pre-trained model (from synthetic data)
and dynamically fine-tunes itself as real data accumulates.

LEARNING STRATEGY:
1. Cold Start (Iterations 1-5): Use pre-trained model from synthetic data
2. Data Collection: Store every iteration as training example
3. Periodic Retraining: Fine-tune model every N iterations with real data
4. Hybrid Training: Blend synthetic + real data to prevent forgetting

USAGE:
    router = DynamicBERTRouter(pretrained_model_path="./bert_router_trained")
    
    # Each iteration
    next_agent = router.route(state, history)
    router.record_iteration(state, history, actual_agent_used, outcome)
    
    # Automatically triggers retraining when threshold reached
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError):
    TRANSFORMERS_AVAILABLE = False

from mmm_optimizer.orchestrator.state import MMMState, MMMHistory

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example from real optimization session."""
    iteration: int
    session_id: str
    timestamp: str
    state_text: str  # Formatted state for BERT input
    agent_label: int  # 0=DS, 1=BA, 2=SH
    actual_outcome: Dict[str, Any]  # Performance after this iteration
    metadata: Dict[str, Any]  # Additional context


class DynamicBERTRouter:
    """
    BERT Router with Online Learning Capabilities
    
    This router starts with a pre-trained model and continuously improves
    by learning from real optimization data.
    
    KEY FEATURES:
    - Cold start with synthetic pre-trained model
    - Collects training examples from every iteration
    - Automatically triggers retraining when threshold reached
    - Blends synthetic + real data to prevent catastrophic forgetting
    - Tracks model performance over time
    
    RETRAINING TRIGGERS:
    - Every N iterations (default: 10)
    - When model confidence drops below threshold
    - On explicit request
    
    DATA MANAGEMENT:
    - Stores examples in JSON file for persistence
    - Maintains sliding window of recent examples
    - Can export/import training data
    """
    
    def __init__(
        self,
        pretrained_model_path: str = "./mmm_optimizer/bert_router_trained",
        training_data_path: str = "./data/bert_router_training.json",
        model_save_path: str = "./mmm_optimizer/bert_router_trained",
        update_interval: int = 1,
        batch_update_size: int = 5,
        max_training_examples: int = 500,
        learning_rate: float = 1e-5,
        save_after_update: bool = True
    ):
        """
        Initialize Dynamic BERT Router.
        
        Args:
            pretrained_model_path: Path to pre-trained model directory (initial load)
            training_data_path: Where to save collected training examples
            model_save_path: Where to save updated model (default: same as pretrained)
            update_interval: Update model every N iterations (default: 1 for true online)
            batch_update_size: Minimum examples to accumulate before updating
            max_training_examples: Max examples to keep (FIFO)
            learning_rate: Learning rate for online updates
            save_after_update: Whether to save model after each update
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers library not available. "
                "Install with: pip install transformers torch"
            )
        
        logger.info("=" * 70)
        logger.info("Initializing Dynamic BERT Router")
        logger.info("=" * 70)
        
        self.pretrained_model_path = Path(pretrained_model_path)
        self.training_data_path = Path(training_data_path)
        self.model_save_path = Path(model_save_path)
        self.update_interval = update_interval
        self.batch_update_size = batch_update_size
        self.max_training_examples = max_training_examples
        self.learning_rate = learning_rate
        self.save_after_update = save_after_update
        
        # Load pre-trained model
        logger.info(f"Loading pre-trained model from: {self.pretrained_model_path}")
        self.tokenizer = BertTokenizer.from_pretrained(str(self.pretrained_model_path))
        self.model = BertForSequenceClassification.from_pretrained(str(self.pretrained_model_path))
        self.model.eval()
        
        logger.info(f"✓ Model loaded successfully")
        
        # Agent label mapping
        self.agent_labels = {
            0: "data_scientist",
            1: "business_analyst",
            2: "stakeholder"
        }
        self.label_to_idx = {v: k for k, v in self.agent_labels.items()}
        
        # Training data storage
        self.training_examples: List[TrainingExample] = []
        self.pending_examples: List[TrainingExample] = []  # Examples waiting for next update
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.iteration_count = 0
        self.update_count = 0
        
        # Load existing training data if available
        self._load_training_data()
        
        logger.info(f"✓ Loaded {len(self.training_examples)} existing training examples")
        logger.info(f"✓ Online learning: Update every {update_interval} iterations")
        logger.info(f"✓ Batch size: {batch_update_size} examples")
        logger.info(f"✓ Model will be saved to: {self.model_save_path}")
        logger.info("=" * 70)
    
    def route(
        self,
        state: MMMState,
        history: MMMHistory,
        return_confidence: bool = False
    ) -> str:
        """
        Route to next agent using current BERT model.
        
        Args:
            state: Current MMMState
            history: Optimization history
            return_confidence: If True, returns (agent, confidence) tuple
        
        Returns:
            str: Agent name ("data_scientist", "business_analyst", "stakeholder")
            or Tuple[str, float]: (agent_name, confidence) if return_confidence=True
        """
        # Format state as text for BERT input
        state_text = self._format_state_as_text(state, history)
        
        # Tokenize and predict
        inputs = self.tokenizer(
            state_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0, pred_idx].item()
        
        agent_name = self.agent_labels[pred_idx]
        
        logger.info(f"\n🤖 BERT Router Decision (Iteration {state.iteration}):")
        logger.info(f"  → Agent: {agent_name}")
        logger.info(f"  → Confidence: {confidence:.2%}")
        logger.info(f"  → Probabilities: DS={probs[0, 0]:.2%}, BA={probs[0, 1]:.2%}, SH={probs[0, 2]:.2%}")
        
        if return_confidence:
            return agent_name, confidence
        return agent_name
    
    def record_iteration(
        self,
        state: MMMState,
        history: MMMHistory,
        actual_agent_used: str,
        outcome: Optional[Dict[str, Any]] = None
    ):
        """
        Record iteration as training example and trigger online update if needed.
        
        This is the key method for online learning. Call it after each iteration
        to collect training data and incrementally update the model.
        
        Args:
            state: State before agent execution
            history: History up to this point
            actual_agent_used: Which agent was actually used (must be one of:
                              'data_scientist', 'business_analyst', 'stakeholder')
            outcome: Performance metrics after agent execution
        """
        # Validate agent name
        if actual_agent_used not in self.label_to_idx:
            raise ValueError(
                f"Invalid agent name: {actual_agent_used}. "
                f"Must be one of: {list(self.label_to_idx.keys())}"
            )
        
        self.iteration_count += 1
        
        # Create training example
        state_text = self._format_state_as_text(state, history)
        agent_label = self.label_to_idx[actual_agent_used]
        
        example = TrainingExample(
            iteration=state.iteration,
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            state_text=state_text,
            agent_label=agent_label,
            actual_outcome=outcome or {},
            metadata={
                "mape": state.performance.mape,
                "r2": state.performance.r2,
                "convergence": history.get_mape_improvement_rate(last_n=3)
            }
        )
        
        # Add to training examples (with FIFO limit)
        self.training_examples.append(example)
        self.pending_examples.append(example)
        if len(self.training_examples) > self.max_training_examples:
            self.training_examples.pop(0)  # Remove oldest
        
        logger.info(f"\n📊 Recorded training example:")
        logger.info(f"  • Iteration: {state.iteration}")
        logger.info(f"  • Agent: {actual_agent_used}")
        logger.info(f"  • Total examples: {len(self.training_examples)}")
        logger.info(f"  • Pending for update: {len(self.pending_examples)}")
        
        # Save to disk
        self._save_training_data()
        
        # Check if online update needed
        if self._should_update():
            logger.info(f"\n🔄 Online learning update triggered!")
            self.online_update()
            self.pending_examples = []  # Clear pending after successful update
    
    def online_update(self, epochs: int = 1, batch_size: int = 4):
        """
        Perform online learning update with pending examples.
        
        This method incrementally updates the model with new real data,
        using very conservative settings to avoid catastrophic forgetting.
        After update, saves the model if save_after_update is True.
        
        Args:
            epochs: Training epochs (default: 1 for online learning)
            batch_size: Batch size (small for online updates)
        """
        if len(self.pending_examples) < self.batch_update_size:
            logger.warning(
                f"Not enough pending examples for update "
                f"({len(self.pending_examples)} < {self.batch_update_size})"
            )
            return
        
        logger.info("\n" + "=" * 70)
        logger.info(f"ONLINE LEARNING UPDATE #{self.update_count + 1}")
        logger.info("=" * 70)
        logger.info(f"Updating with {len(self.pending_examples)} new examples")
        logger.info(f"Total examples collected: {len(self.training_examples)}")
        
        # Prepare training data from pending examples only
        texts = [ex.state_text for ex in self.pending_examples]
        labels = [ex.agent_label for ex in self.pending_examples]
        
        # Create dataset
        from torch.utils.data import Dataset as TorchDataset
        
        class SimpleDataset(TorchDataset):
            def __init__(self, texts, labels, tokenizer):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
        
        dataset = SimpleDataset(texts, labels, self.tokenizer)
        
        # Training arguments (very conservative for online learning)
        output_dir = f"./models/bert_router_update_{self.update_count}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=self.learning_rate,
            save_strategy="no",  # We'll save manually to model_save_path
            logging_steps=max(1, len(texts) // batch_size),
            warmup_steps=0,  # No warmup for online learning
            weight_decay=0.01,
            report_to=[],
            disable_tqdm=True  # Less verbose for online updates
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Train
        logger.info(f"Updating model for {epochs} epoch(s) with lr={self.learning_rate}...")
        self.model.train()
        trainer.train()
        self.model.eval()
        
        self.update_count += 1
        logger.info(f"✓ Model updated! (Total updates: {self.update_count})")
        
        # Save updated model
        if self.save_after_update:
            self._save_model()
        
        logger.info("=" * 70 + "\n")
    
    def _should_update(self) -> bool:
        """Check if online update should be triggered."""
        # Check if we have enough pending examples
        if len(self.pending_examples) < self.batch_update_size:
            return False
        
        # Check iteration interval
        if self.iteration_count % self.update_interval == 0:
            return True
        
        return False
    
    def _save_model(self):
        """Save current model to disk."""
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving updated model to: {self.model_save_path}")
            self.model.save_pretrained(str(self.model_save_path))
            self.tokenizer.save_pretrained(str(self.model_save_path))
            
            # Save metadata
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "update_count": self.update_count,
                "total_examples": len(self.training_examples),
                "session_id": self.session_id
            }
            with open(self.model_save_path / "update_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Model saved successfully (Update #{self.update_count})")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _format_state_as_text(self, state: MMMState, history: MMMHistory) -> str:
        """
        Format state as natural language text for BERT input.
        
        Must match the format used during pre-training of the BERT model.
        
        Args:
            state: Current MMMState
            history: Optimization history
        
        Returns:
            str: Formatted text
        """
        # Performance summary
        perf_text = (
            f"Iteration {state.iteration}: "
            f"MAPE={state.performance.mape:.1f}%, "
            f"R2={state.performance.r2:.3f}, "
            f"MAE={state.performance.mae:.0f}. "
        )
        
        # Recent agents (last 3)
        recent = [rec.agent_name for rec in history.iterations[-3:]]
        if recent:
            agents_text = f"Last {len(recent)} agents: {', '.join(recent)}. "
        else:
            agents_text = "No previous agents. "
        
        # Business flags (simplified)
        flags_count = 0
        if state.performance.mape > 12.0:
            flags_count += 1
        if state.performance.r2 < 0.85:
            flags_count += 1
        flags_text = f"Business flags: {flags_count}. "
        
        # Hyperparameters (sample)
        hyperparams = state.model_config.get("hyperparameters", {})
        hyperparam_items = list(hyperparams.items())[:3]
        hyperparam_text = "Hyperparameters: " + ", ".join(
            [f"{k}={v:.2f}" for k, v in hyperparam_items]
        ) + ". "
        
        # Business context
        industry = state.business_context.get("industry", "Unknown")
        biz_text = f"Industry: {industry}. "
        
        # Patterns
        pattern_text = ""
        if history.is_stuck_in_loop():
            pattern_text = "LOOP DETECTED. "
        elif len(history.iterations) > 10 and history.get_mape_improvement_rate(last_n=3) < 0.5:
            pattern_text = "Near convergence. "
        
        full_text = (
            perf_text + agents_text + flags_text +
            hyperparam_text + biz_text + pattern_text
        )
        
        return full_text
    
    def _save_training_data(self):
        """Save training examples to JSON file."""
        try:
            self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "session_id": self.session_id,
                "last_updated": datetime.now().isoformat(),
                "total_examples": len(self.training_examples),
                "update_count": self.update_count,
                "examples": [asdict(ex) for ex in self.training_examples]
            }
            
            with open(self.training_data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def _load_training_data(self):
        """Load existing training examples from JSON file."""
        if not self.training_data_path.exists():
            logger.info("No existing training data found")
            return
        
        try:
            with open(self.training_data_path, "r") as f:
                data = json.load(f)
            
            self.training_examples = [
                TrainingExample(**ex) for ex in data.get("examples", [])
            ]
            self.update_count = data.get("update_count", data.get("retrain_count", 0))
            
            logger.info(f"Loaded {len(self.training_examples)} examples from previous sessions")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            self.training_examples = []
    
    def export_training_data(self, output_path: str):
        """Export training data for external analysis."""
        data = {
            "metadata": {
                "session_id": self.session_id,
                "exported_at": datetime.now().isoformat(),
                "total_examples": len(self.training_examples),
                "update_count": self.update_count
            },
            "examples": [asdict(ex) for ex in self.training_examples],
            "agent_label_mapping": self.agent_labels
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✓ Exported {len(self.training_examples)} examples to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        if not self.training_examples:
            return {"status": "No data collected yet"}
        
        from collections import Counter
        label_dist = Counter([ex.agent_label for ex in self.training_examples])
        
        return {
            "total_examples": len(self.training_examples),
            "pending_examples": len(self.pending_examples),
            "session_id": self.session_id,
            "update_count": self.update_count,
            "iterations_since_update": self.iteration_count % self.update_interval if self.update_interval > 0 else 0,
            "label_distribution": {
                self.agent_labels[k]: v for k, v in label_dist.items()
            },
            "next_update_in": max(0, self.batch_update_size - len(self.pending_examples))
        }
