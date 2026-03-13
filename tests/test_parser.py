import sys
from pathlib import Path

# ensure imports work
sys.path.insert(0, str(Path.cwd()))

from src.parser import parse_supply_chain_text


def test_parse_nodes():
    text = "Node A. Node B."
    entities = parse_supply_chain_text(text)
    assert len(entities["nodes"]) == 2
    assert entities["nodes"][0]["id"] == "A"
    assert entities["nodes"][1]["id"] == "B"


def test_parse_products():
    text = "Product P1. Product P2."
    entities = parse_supply_chain_text(text)
    assert len(entities["products"]) == 2
    assert entities["products"][0]["id"] == "P1"


def test_parse_suppliers():
    text = "Supplier S1 in A supplies P with capacity 10."
    entities = parse_supply_chain_text(text)
    assert len(entities["suppliers"]) == 1
    s = entities["suppliers"][0]
    assert s["id"] == "S1"
    assert s["node"] == "A"
    assert s["product"] == "P"
    assert s["capacity"] == 10


def test_parse_consumers():
    text = "Consumer C1 in B demands Q with capacity 5."
    entities = parse_supply_chain_text(text)
    assert len(entities["consumers"]) == 1
    c = entities["consumers"][0]
    assert c["id"] == "C1"
    assert c["node"] == "B"
    assert c["product"] == "Q"
    assert c["capacity"] == 5


def test_parse_transport_links():
    text = "Transport T1 from A to B for P with capacity 20."
    entities = parse_supply_chain_text(text)
    assert len(entities["transport_links"]) == 1
    t = entities["transport_links"][0]
    assert t["id"] == "T1"
    assert t["origin"] == "A"
    assert t["destination"] == "B"
    assert t["product"] == "P"
    assert t["capacity"] == 20


def test_parse_technologies():
    text = "Technology Tech1 in A with capacity 15."
    entities = parse_supply_chain_text(text)
    assert len(entities["technologies"]) == 1
    tech = entities["technologies"][0]
    assert tech["id"] == "Tech1"
    assert tech["node"] == "A"
    assert tech["capacity"] == 15
    assert tech["yield_coefficients"] == {}


def test_parse_bids():
    text = "Bid B1 by S1 (supplier) for P price 1.5 quantity 10."
    entities = parse_supply_chain_text(text)
    assert len(entities["bids"]) == 1
    b = entities["bids"][0]
    assert b["id"] == "B1"
    assert b["owner_id"] == "S1"
    assert b["owner_type"] == "supplier"
    assert b["product_id"] == "P"
    assert b["price"] == 1.5
    assert b["quantity"] == 10


def test_parse_negative_bid():
    text = "Bid B2 by C1 (consumer) for Q price -2.0 quantity 5."
    entities = parse_supply_chain_text(text)
    assert len(entities["bids"]) == 1
    b = entities["bids"][0]
    assert b["price"] == -2.0


def test_parse_multiple_entities():
    text = """
    Node N1. Product Prod. Supplier Sup in N1 supplies Prod with capacity 100.
    Consumer Con in N1 demands Prod with capacity 50.
    Bid Bid1 by Sup (supplier) for Prod price 10.0 quantity 50.
    """
    entities = parse_supply_chain_text(text)
    assert len(entities["nodes"]) == 1
    assert len(entities["products"]) == 1
    assert len(entities["suppliers"]) == 1
    assert len(entities["consumers"]) == 1
    assert len(entities["bids"]) == 1


def test_parse_empty_text():
    entities = parse_supply_chain_text("")
    for key in entities:
        assert entities[key] == []


def test_case_insensitive():
    text = "NODE A. PRODUCT P. SUPPLIER S IN A SUPPLIES P WITH CAPACITY 10."
    entities = parse_supply_chain_text(text)
    assert len(entities["nodes"]) == 1
    assert len(entities["products"]) == 1
    assert len(entities["suppliers"]) == 1


def test_parse_with_llm_enabled():
    """Ensure LLM provider is used when `use_llm=True`."""
    from src.llm_adapter import LLMProviderRegistry, RuleBasedProvider
    from unittest.mock import Mock

    # Create a provider whose parser returns a distinctive result
    def fake_parse(text):
        return {
            "nodes": [{"id": "LLM_NODE", "name": "LLM_NODE"}],
            "products": [],
            "suppliers": [],
            "consumers": [],
            "transport_links": [],
            "technologies": [],
            "bids": [],
        }

    provider = RuleBasedProvider(
        intent_router=Mock(),
        parse_function=fake_parse,
        generate_function=lambda mode, ctx: "",
    )
    registry = LLMProviderRegistry.get_instance()
    registry.set_provider(provider)

    entities = parse_supply_chain_text("irrelevant text", use_llm=True)
    assert len(entities["nodes"]) == 1
    assert entities["nodes"][0]["id"] == "LLM_NODE"

    # Clean up registry state
    registry.reset()


def test_parse_llm_fallback_on_error():
    """If LLM parser fails, fall back to rule-based parsing."""
    from src.llm_adapter import LLMProviderRegistry, RuleBasedProvider
    from unittest.mock import Mock

    def broken_parse(text):
        raise RuntimeError("parser crashed")

    provider = RuleBasedProvider(
        intent_router=Mock(),
        parse_function=broken_parse,
        generate_function=lambda mode, ctx: "",
    )
    registry = LLMProviderRegistry.get_instance()
    registry.set_provider(provider)

    # Should not raise despite LLM parser error
    entities = parse_supply_chain_text("Node Z", use_llm=True)
    assert entities["nodes"] and entities["nodes"][0]["id"] == "Z"

    registry.reset()
