import logging

from fastapi import APIRouter, HTTPException, status

from ..repositories import repository
from ..schemas import UserCreate, UserUpdate, UserProfile

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/users", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate):
    user_id = repository.create_user(
        full_name=payload.full_name
    )
    user = repository.get_user(user_id)
    return user


@router.get("/users/{user_id}", response_model=UserProfile)
def get_user(user_id: str):
    user = repository.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}")
def update_user(user_id: str, payload: UserUpdate):
    ok = repository.update_user(
        user_id=user_id,
        full_name=payload.full_name,
        email=payload.email,
        phone=payload.phone,
        language=payload.language,
        timezone=payload.timezone,
        consent_accepted=payload.consent_accepted,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok"}
