"""Synthetic prompt workload generator for benchmark runs."""

from __future__ import annotations

import random
from typing import Literal

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkloadGenerator:
    """Generate synthetic prompts at specified token lengths."""
    
    # Token approximations (words to tokens ratio varies by model)
    AVG_TOKENS_PER_WORD = 1.3
    
    # Sample text corpus for synthetic generation
    SAMPLE_WORDS = [
        "machine", "learning", "artificial", "intelligence", "neural", "network",
        "deep", "learning", "model", "training", "inference", "optimization",
        "performance", "benchmark", "throughput", "latency", "gpu", "cuda",
        "tensor", "matrix", "computation", "parallel", "processing", "memory",
        "bandwidth", "efficiency", "scalability", "distributed", "system",
        "architecture", "algorithm", "data", "structure", "analysis",
        "research", "development", "production", "deployment", "serving",
        "api", "gateway", "routing", "load", "balancing", "autoscaling",
        "kubernetes", "container", "docker", "microservices", "cloud",
        "compute", "accelerator", "hardware", "software", "engineering",
        "computer", "vision", "natural", "language", "processing", "speech",
        "recognition", "generation", "classification", "regression",
        "clustering", "embedding", "vector", "database", "search",
    ]
    
    def __init__(self, tokenizer_name: str | None = None) -> None:
        """Initialize workload generator.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name (if None, uses word approximation)
        """
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info("Tokenizer loaded", name=tokenizer_name)
            except Exception as e:
                logger.warning(
                    "Failed to load tokenizer, using word approximation",
                    name=tokenizer_name,
                    error=str(e),
                )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text."""
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: word-based estimation
        words = len(text.split())
        return int(words * self.AVG_TOKENS_PER_WORD)
    
    def generate_prompt(
        self,
        target_tokens: int,
        distribution: Literal["fixed", "uniform", "normal"] = "fixed",
        variance: float = 0.1,
    ) -> str:
        """Generate a synthetic prompt targeting specific token count.
        
        Args:
            target_tokens: Target number of tokens
            distribution: How to vary the actual token count
            variance: Variance factor for non-fixed distributions
            
        Returns:
            Synthetic prompt text
        """
        # Adjust target based on distribution
        if distribution == "uniform":
            actual_target = int(target_tokens * random.uniform(1 - variance, 1 + variance))
        elif distribution == "normal":
            actual_target = int(random.gauss(target_tokens, target_tokens * variance))
        else:
            actual_target = target_tokens
        
        actual_target = max(10, actual_target)  # Minimum 10 tokens
        
        # Estimate words needed
        target_words = int(actual_target / self.AVG_TOKENS_PER_WORD)
        
        # Build prompt
        words = []
        current_tokens = 0
        
        # Add instruction prefix
        prefixes = [
            "Explain the following concept in detail:",
            "Write a comprehensive analysis about:",
            "Describe the process of:",
            "Compare and contrast these approaches to:",
            "Provide a detailed explanation of:",
        ]
        prefix = random.choice(prefixes)
        words.extend(prefix.split())
        current_tokens = self.estimate_tokens(" ".join(words))
        
        # Add random words until we reach target
        while current_tokens < actual_target:
            word = random.choice(self.SAMPLE_WORDS)
            words.append(word)
            current_tokens = self.estimate_tokens(" ".join(words))
        
        prompt = " ".join(words)
        
        # Verify and log actual token count
        actual_tokens = self.estimate_tokens(prompt)
        logger.debug(
            "Generated prompt",
            target=target_tokens,
            actual=actual_tokens,
            words=len(words),
        )
        
        return prompt
    
    def generate_batch(
        self,
        target_tokens: int,
        batch_size: int,
        distribution: Literal["fixed", "uniform", "normal"] = "fixed",
    ) -> list[str]:
        """Generate a batch of prompts.
        
        Args:
            target_tokens: Target tokens per prompt
            batch_size: Number of prompts to generate
            distribution: Token count distribution
            
        Returns:
            List of prompt strings
        """
        return [
            self.generate_prompt(target_tokens, distribution)
            for _ in range(batch_size)
        ]
    
    def generate_variable_batch(
        self,
        token_range: tuple[int, int],
        batch_size: int,
    ) -> list[str]:
        """Generate batch with variable-length prompts.
        
        Args:
            token_range: (min_tokens, max_tokens)
            batch_size: Number of prompts
            
        Returns:
            List of prompts with varying lengths
        """
        min_tokens, max_tokens = token_range
        prompts = []
        for _ in range(batch_size):
            target = random.randint(min_tokens, max_tokens)
            prompts.append(self.generate_prompt(target, distribution="fixed"))
        return prompts
    
    def generate_dataset(
        self,
        prompt_lengths: list[int],
        batch_sizes: list[int],
        samples_per_config: int = 1,
    ) -> dict[tuple[int, int], list[list[str]]]:
        """Generate full benchmark dataset.
        
        Args:
            prompt_lengths: List of target token lengths
            batch_sizes: List of batch sizes
            samples_per_config: Number of batches per configuration
            
        Returns:
            Dictionary mapping (prompt_length, batch_size) to list of batches
        """
        dataset: dict[tuple[int, int], list[list[str]]] = {}
        
        for length in prompt_lengths:
            for batch_size in batch_sizes:
                batches = []
                for _ in range(samples_per_config):
                    batch = self.generate_batch(length, batch_size)
                    batches.append(batch)
                dataset[(length, batch_size)] = batches
        
        logger.info(
            "Generated benchmark dataset",
            configurations=len(dataset),
            total_batches=sum(len(b) for b in dataset.values()),
        )
        
        return dataset


def create_workload_generator(
    tokenizer: str | None = None,
) -> WorkloadGenerator:
    """Factory function to create workload generator."""
    return WorkloadGenerator(tokenizer_name=tokenizer)
