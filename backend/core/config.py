from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field("Assistant Demo", env="APP_NAME")
    app_version: str = Field("0.1.0", env="APP_VERSION")

    database_url: str = Field("sqlite:///./data/app.db", env="DATABASE_URL")
    timezone: str = Field("UTC", env="TIMEZONE")

    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    decoder_use_openai: bool = Field(default=False, env="DECODER_USE_OPENAI")

    intent_model_path: str = Field("./models/intention/robertuito_finetuned", env="INTENT_MODEL_PATH")
    sentiment_model_path: str = Field("./models/sentiment/dccuchile_bert_spanish_finetuned", env="SENTIMENT_MODEL_PATH")
    emotion_model_path: str = Field(
        "./models/emotion/pysentimiento_robertuito_sentiment_analysis_finetuned", env="EMOTION_MODEL_PATH"
    )
    ner_model_path: str = Field("./models/ner/spacy/model_es_ner", env="NER_MODEL_PATH")

    flow_use_llm: bool = Field(default=True, env="FLOW_USE_LLM")
    flow_llm_model: str = Field(default="gpt-4o-mini", env="FLOW_LLM_MODEL")
    flow_llm_threshold: float = Field(default=0.75, env="FLOW_LLM_THRESHOLD")
    decoder_max_tokens: int = Field(default=200, env="DECODER_MAX_TOKENS")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
