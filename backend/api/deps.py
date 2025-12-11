from fastapi import Depends, HTTPException, Request

from ..services.orchestrator import ConversationOrchestrator


def get_orchestrator(request: Request) -> ConversationOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orch
