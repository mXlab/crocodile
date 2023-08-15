from typing import Any, Dict, Optional


def flatten_dict(obj: Dict[str, Any], parent: Optional[str] = None) -> Dict[str, Any]:
    """Flatten dict"""
    result = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            if parent is not None:
                k = f"{parent}.{k}"
            result.update(flatten_dict(v, parent=k))
        else:
            if parent is not None:
                k = f"{parent}.{k}"
            result[k] = v
    return result
