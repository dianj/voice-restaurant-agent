"""LM Studio model management through its REST API (list, load, unload, switch)."""
import json
import threading
import urllib.error
import urllib.request

from .config import ALLOWED_MODELS, LMSTUDIO_ROOT, LOAD_CONTEXT_LENGTH, MODEL_LOAD_TIMEOUT, MODEL_NAME

# The hardware fits only one model in memory, so the active model is global (not per session):
# switching unloads the current model in LM Studio and then loads the selected one.
model_lock = threading.Lock()
model_state = {"active": MODEL_NAME, "switching": False, "target": None, "error": None}

def lmstudio_request(method: str, path: str, body: dict = None, timeout: float = 10):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{LMSTUDIO_ROOT}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LM Studio {method} {path} failed ({e.code}): {detail}") from e

def get_lmstudio_models() -> dict:
    """Map model key -> LM Studio model info (includes loaded_instances)."""
    return {m["key"]: m for m in lmstudio_request("GET", "/api/v1/models").get("models", [])}

def list_available_models():
    try:
        lm_models = get_lmstudio_models()
    except Exception:
        lm_models = {}
    result = []
    for model_id, label in ALLOWED_MODELS:
        info = lm_models.get(model_id)
        result.append({
            "id": model_id,
            "label": label,
            "installed": info is not None if lm_models else None,
            "loaded": bool(info and info.get("loaded_instances")) if lm_models else None,
            "vision": bool(info and info.get("capabilities", {}).get("vision")) if lm_models else None,
        })
    return result

def activate_model(target: str):
    """Unload every other allowed model, then make sure `target` is loaded. Blocking."""
    lm_models = get_lmstudio_models()
    if target not in lm_models:
        raise RuntimeError(f"Model '{target}' is not installed in LM Studio.")
    allowed_ids = {model_id for model_id, _ in ALLOWED_MODELS}
    for model_id in allowed_ids - {target}:
        for instance in (lm_models.get(model_id) or {}).get("loaded_instances", []):
            lmstudio_request("POST", "/api/v1/models/unload", {"instance_id": instance["id"]}, timeout=120)
    if not lm_models[target].get("loaded_instances"):
        lmstudio_request(
            "POST", "/api/v1/models/load",
            {"model": target, "context_length": LOAD_CONTEXT_LENGTH},
            timeout=MODEL_LOAD_TIMEOUT
        )

def switch_model(target: str):
    """Switch the global active model; on failure try to restore the previous one."""
    with model_lock:
        previous = model_state["active"]
        model_state.update(switching=True, target=target, error=None)
        try:
            activate_model(target)
            model_state["active"] = target
            return {"status": "success", "active": target}
        except Exception as e:
            model_state["error"] = str(e)
            try:
                activate_model(previous)
                restored = f" Restored '{previous}'."
            except Exception as restore_error:
                restored = f" Could not restore '{previous}': {restore_error}"
            return {"status": "error", "message": f"{e}{restored}", "active": previous}
        finally:
            model_state.update(switching=False, target=None)

def ensure_default_model_on_startup():
    try:
        switch_model(MODEL_NAME)
    except Exception:
        pass  # LM Studio may not be running yet; the UI reports connection errors per request.
