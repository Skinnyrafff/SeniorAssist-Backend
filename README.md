
  # Senior Assist V1

  Asistente virtual para adultos mayores. Permite recibir texto, detectar intención, sentimiento, emoción y entidades, y responder de forma empática.

  ## Instalación rápida

  1. Instala dependencias:
    ```bash
    pip install -r requirements.txt
    ```
  2. Copia `.env.example` a `.env` y completa tus variables (API key, rutas de modelos, etc).

  ## Uso

  1. Inicia el backend:
    ```bash
    uvicorn backend.main:app --reload --port 8000
    ```
  2. Inicia el frontend:
    ```bash
    streamlit run frontend/streamlit_app.py
    ```

  ## Estructura básica

  - `backend/`: API y lógica principal (FastAPI)
  - `frontend/`: Interfaz de usuario (Streamlit)
  - `models/`: Modelos locales (no se suben a GitHub)
  - `scripts/`: utilidades y pruebas

  ## Notas

  - No subas archivos de modelos ni `.env` a GitHub.
  - El sistema puede funcionar en modo demo sin API key.

  ---
  Primera versión pública. Para dudas o mejoras, abre un issue o pull request.
