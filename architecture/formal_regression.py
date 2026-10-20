"""Importable ABI scaffold for independent hosted assertion-RED controls.

This deliberately implements none of the accepted collector/core behavior.
A literal empty result is not admission, native evidence or a successful core.
Working code may follow only qualified hosted RED and distinct technical review.
"""
Record = dict[str, object]
RequestContext = Record
Policy = Record
TargetStatement = Record
InventoryResult = Record


def decode_request(raw: bytes) -> RequestContext:
    return {}


def decode_policy(raw: bytes, expected_sha256: str) -> Policy:
    return {}


def request_key(context: RequestContext) -> str:
    return ""


def content_inventory(records: list[TargetStatement], policy: Policy,
                      membership: Record, acceptances: Record) -> InventoryResult:
    return {"core": None, "formal_inventory_sha256": None,
            "diagnostic_inventory_sha256": None, "issues": []}


def inventory_sha256(core: Record) -> str:
    return ""
