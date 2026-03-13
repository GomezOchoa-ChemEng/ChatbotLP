"""Integration guide for LLM layer in the supply chain chatbot.

This document shows how to use the new LLM integration layer to extend the
chatbot with real LLM implementations while maintaining the deterministic
rule-based system as the default.

Overview
=========

The LLM integration layer (llm_interfaces.py + llm_adapter.py) defines three
core interfaces:

1. IntentClassifier: Classify user intent from natural language
2. SupplyChainParser: Extract entities from problem descriptions
3. ExplanationGenerator: Generate responses in classroom modes (hint/guided/full)

The system includes:
- Mock implementations (placeholder behavior, no external API calls)
- Rule-based adapters (wrap existing deterministic algorithms)
- A provider registry (switch implementations at runtime)

Design Principles
==================

1. **Interface Segregation**: Clean abstract interfaces reduce coupling
2. **Provider Pattern**: Swap implementations without modifying core code
3. **Backward Compatible**: Existing code continues to work unchanged
4. **Deterministic Defaults**: Rule-based system is the default
5. **LLM Optional**: Real LLMs are plugged in, not baked in
6. **Testable**: All implementations can be mocked for testing

Usage Examples
==============

Example 1: Use the default rule-based system
---------------------------------------------

from src.chatbot_engine import run_chatbot_session
from src.state_manager import StateManager

# StateManager handles problem persistence
state_mgr = StateManager()
problem_state = state_mgr.new_problem()

# This uses rule-based intent detection, parsing, and explanation generation
# by default. No LLM layer configuration needed. You can also explicitly
# disable any LLM behaviour by passing ``use_llm=False`` (the default).
user_message = "Add node N1 and product P1"
response_dict = run_chatbot_session(problem_state, user_message, "hint")

print(response_dict["response"])

# If an LLM provider is registered and you want to invoke it without
# installing a rule-based router, specify the flag:
# response_dict = run_chatbot_session(problem_state, user_message, "hint", use_llm=True)


Example 2: Use mock LLM implementations for testing
----------------------------------------------------

from src.llm_adapter import MockLLMProvider, LLMProviderRegistry
from src.llm_interfaces import IntentClassifier, SupplyChainParser

# Register mock implementations
registry = LLMProviderRegistry.get_instance()
registry.set_provider(MockLLMProvider())

# Now get implementations from the registry
classifier = registry.get_intent_classifier()
intent = classifier.classify("solve the model")
# Returns: "problem_formulation" (mock default)

parser = registry.get_parser()
entities = parser.parse("Node N1 supplies Product P1")
# Returns: {"nodes": [], "products": [], ...} (empty mock)

generator = registry.get_explanation_generator()
response = generator.generate("hint", {})
# Returns: "[Mock LLM response in hint mode - placeholder]"


Example 3: Switch to rule-based via registry (optional, for explicit control)
---------------------------------------------------------------------

from src.llm_adapter import RuleBasedProvider, LLMProviderRegistry
from src.chatbot_engine import IntentRouter
from src.parser import parse_supply_chain_text
from src.response_generator import ResponseGenerator

# Create instances of rule-based components
intent_router = IntentRouter()
response_gen = ResponseGenerator()

# Wrap them in a provider
provider = RuleBasedProvider(
    intent_router=intent_router,
    parse_function=parse_supply_chain_text,
    generate_function=response_gen.generate_response,
)

# Register the provider
registry = LLMProviderRegistry.get_instance()
registry.set_provider(provider)

# Now the registry uses rule-based implementations
classifier = registry.get_intent_classifier()
intent = classifier.classify("Check the problem state")
# Returns: "validation" (detected by IntentRouter)


Example 4: Implement a custom LLM provider (future extension)
--------------------------------------------------------------

from src.llm_interfaces import (
    IntentClassifier,
    SupplyChainParser,
    ExplanationGenerator,
    LLMProvider,
)
import openai  # or anthropic, litellm, etc


class GPTIntentClassifier(IntentClassifier):
    """Intent classifier using GPT-4."""
    
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def classify(self, text: str) -> str:
        """Use GPT-4 to classify intent."""
        if not text or not text.strip():
            raise ValueError("Intent classifier received empty text")
        
        prompt = f"""Classify the user's intent as one of:
        - problem_formulation: describing problem structure
        - validation: checking the problem for issues
        - solve: requesting optimization
        - theorem_check: asking about theoretical assumptions
        - scenario: requesting what-if analysis
        - explanation: asking for help or guidance
        
        User message: {text}
        
        Respond with only the intent name."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        
        intent = response.choices[0].message.content.strip().lower()
        valid_intents = [
            "problem_formulation",
            "validation",
            "solve",
            "theorem_check",
            "scenario",
            "explanation",
        ]
        return intent if intent in valid_intents else "problem_formulation"


class GPTSupplyChainParser(SupplyChainParser):
    """Parser using GPT-4 for entity extraction."""
    
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def parse(self, text: str) -> dict:
        """Use GPT-4 to extract supply chain entities."""
        if not text or not text.strip():
            raise ValueError("Parser received empty text")
        
        prompt = f"""Extract supply chain entities from this text.
        Return JSON with keys: nodes, products, suppliers, consumers,
        transport_links, technologies, bids.
        Each entity should have relevant fields (id, name, capacity, etc.).
        
        Text: {text}
        
        Return only valid JSON, no other text."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback to empty entities if parsing fails
            return {
                "nodes": [],
                "products": [],
                "suppliers": [],
                "consumers": [],
                "transport_links": [],
                "technologies": [],
                "bids": [],
            }


class GPTExplanationGenerator(ExplanationGenerator):
    """Explanation generator using GPT-4."""
    
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, mode: str, context: dict) -> str:
        """Use GPT-4 to generate explanations."""
        if mode not in ("hint", "guided", "full"):
            raise ValueError(f"Invalid mode: {mode}")
        
        mode_instructions = {
            "hint": "Provide minimal guidance without revealing solutions. Ask guiding questions.",
            "guided": "Provide step-by-step assistance with partial results.",
            "full": "Provide complete solution with detailed explanations.",
        }
        
        problem_str = str(context.get("problem_state", ""))
        validation_str = str(context.get("validation_result", ""))
        
        prompt = f"""You are a teaching assistant for supply chain optimization.
        
        Mode: {mode} - {mode_instructions[mode]}
        
        Current problem state:
        {problem_str}
        
        Validation results:
        {validation_str}
        
        Generate an appropriate response for the student."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        
        return response.choices[0].message.content


class GPTLLMProvider(LLMProvider):
    """Provider integrating all three GPT-based components."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self._classifier = None
        self._parser = None
        self._generator = None
    
    def get_intent_classifier(self) -> IntentClassifier:
        if self._classifier is None:
            self._classifier = GPTIntentClassifier(self.api_key)
        return self._classifier
    
    def get_parser(self) -> SupplyChainParser:
        if self._parser is None:
            self._parser = GPTSupplyChainParser(self.api_key)
        return self._parser
    
    def get_explanation_generator(self) -> ExplanationGenerator:
        if self._generator is None:
            self._generator = GPTExplanationGenerator(self.api_key)
        return self._generator


# Usage:
import os
gpt_provider = GPTLLMProvider(api_key=os.getenv("OPENAI_API_KEY"))
registry = LLMProviderRegistry.get_instance()
registry.set_provider(gpt_provider)

# Now all classifications, parsing, and explanations use GPT-4
classifier = registry.get_intent_classifier()
intent = classifier.classify("Add a supplier at node N1")


Testing With Providers
=======================

from unittest.mock import Mock
from src.llm_adapter import RuleBasedProvider, LLMProviderRegistry

# For unit testing, you can use mocks:
mock_router = Mock(detect_intent=Mock(return_value="solve"))
mock_parse = Mock(return_value={"nodes": [], "products": [], ...})
mock_generate = Mock(return_value="Test response")

provider = RuleBasedProvider(
    intent_router=mock_router,
    parse_function=mock_parse,
    generate_function=mock_generate,
)

registry = LLMProviderRegistry.get_instance()
registry.set_provider(provider)

# Test your code without any external API calls
classifier = registry.get_intent_classifier()
assert classifier.classify("test") == "solve"


Architecture Diagram
====================

User Input
    ↓
Chatbot Engine  (imports: parser, intent router, response generator)
    ↓
    ├→ Intent Router (or IntentClassifier from registry)
    │      ↓
    │   [Rule-based, Mock, or LLM]
    │      ↓
    │   Intent detected
    │
    ├→ Parser (or SupplyChainParser from registry)
    │      ↓
    │   [Rule-based, Mock, or LLM]
    │      ↓
    │   Entities extracted
    │
    └→ Response Generator (or ExplanationGenerator from registry)
           ↓
        [Rule-based, Mock, or LLM]
           ↓
        Response generated

Mathematical Core (solver, model builder, theorem checker, scenario engine)
    ↓
User-facing response


Key Design Decisions
====================

1. **Why Interfaces Instead of Direct LLM Calls?**
   - Clean separation between core logic and external services
   - Easy to test without network/API dependencies
   - Easy to swap providers without rewriting code
   - Clear contracts for what each component must do

2. **Why Three Separate Interfaces?**
   - Intent classification, entity extraction, and explanation generation
     have different failure modes and requirements
   - Allows mixing implementations (e.g., rule-based parser + LLM classifier)
   - Easier to test and debug independently

3. **Why a Registry Pattern?**
   - Centralized configuration
   - Runtime switching without restarting
   - Testable (can set registry in test setup)
   - No global state coupling in individual modules

4. **Why Keep Rule-Based as Default?**
   - Deterministic for classroom reproducibility
   - No external API dependencies
   - No cost or latency issues
   - Works offline
   - LLM can be added when needed

Deployment Scenarios
====================

Scenario 1: Classroom (Offline, Deterministic)
- Use default rule-based system
- No API keys required
- Fully reproducible results
- No internet dependency

Scenario 2: Classroom With Assistant
- Deploy with real LLM (GPT-4, Claude, etc.)
- Use provider pattern to swap implementations
- Keep rule-based as fallback
- Costs managed by institution

Scenario 3: Research Use
- Benchmark rule-based vs. LLM-based
- Use mock provider for rapid testing
- Switch providers to compare results
- Log both classical and LLM decisions

Scenario 4: Production APIs
- Expose chatbot as REST API
- Use provider pattern for LLM selection
- Cache responses where deterministic
- Monitor LLM failures and fall back to rule-based


Migration Path (If Desired)
===========================

1. Current state: All deterministic, no LLM integration
   - Works as is, no changes needed

2. Optional Phase 1: Add mock implementations
   - Set MockLLMProvider in registry for testing
   - Verify integration without real API calls

3. Optional Phase 2: Add real LLM provider
   - Implement GPTLLMProvider (or similar)
   - Set in registry for selected classroom/use
   - Keep rule-based as default

4. Optional Phase 3: Hybrid approach
   - Use rule-based intent router (fast, deterministic)
   - Use LLM for explanation generation (more natural)
   - Fine-tune which components benefit from LLM


Summary
=======

The LLM integration layer:
- Defines three clean interfaces (intent, parser, generator)
- Provides mock implementations (no external calls)
- Includes adapters wrapping rule-based code
- Uses registry pattern for runtime switching
- Maintains full backward compatibility
- Enables future LLM integration without major refactoring

The system is ready for either:
1. Continued rule-based operation (current state)
2. Seamless LLM integration when desired

No modifications to the mathematical core or existing modules are required.
"""
