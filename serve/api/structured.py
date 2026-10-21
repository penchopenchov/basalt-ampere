"""JSON response formats: the request is checked here; the engine enforces the schema with a grammar."""
import json


def check_format(response_format) -> bool:
    """True for a JSON response format (json_object or json_schema), False for plain text. ValueError for a request
    the server cannot take."""
    if response_format is None:
        return False
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return False
    if kind == "json_object":
        return True
    if kind != "json_schema":
        raise ValueError("response_format.type must be text, json_object or json_schema")
    spec = response_format.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise ValueError("response_format.json_schema needs a schema object")
    if not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("response_format.json_schema needs a name")
    if "strict" in spec and not isinstance(spec["strict"], bool):
        raise ValueError("response_format.json_schema.strict must be boolean")
    if spec["schema"].get("type") != "object":
        raise ValueError("response_format schema must have type object at its root")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(spec["schema"])
    return True


def anthropic_format(req: dict):
    """The chat path's response_format for an Anthropic request's `output_config.format` (or the older top-level
    `output_format`), or None for plain text. ValueError for a request the server cannot take."""
    out = req.get("output_config")
    fmt = out.get("format") if isinstance(out, dict) else None
    where = "output_config.format"
    if fmt is None and req.get("output_format") is not None:
        fmt, where = req["output_format"], "output_format"
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise ValueError(f"{where}.type must be json_schema")
    response_format = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt.get("schema")}}
    try:
        check_format(response_format)
    except ValueError as e:
        raise ValueError(str(e).replace("response_format.json_schema", where).replace("response_format", where)) from None
    return response_format
