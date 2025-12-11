# Documentación Técnica del Backend

## Índice
1. Introducción
2. Arquitectura General
3. Componentes Principales
   - Configuración y Entorno
   - Modelos de Base de Datos
   - Repositorio y Persistencia
   - Adaptadores y Predictores ML
   - Orquestador de Conversaciones
   - API REST (endpoints)
   - Extracción de Recordatorios
4. Flujos Críticos
   - Conversación y Decisión de Flujos
   - Emergencias
   - Recordatorios
   - Métricas de Salud
5. Esquemas y Validaciones
6. Seguridad
7. Dependencias
8. Notas de Extensión y Mantenimiento

---

## 1. Introducción
El backend de Senior Assist V1 es una API REST desarrollada en Python usando FastAPI, orientada a la asistencia de adultos mayores mediante procesamiento de lenguaje natural, gestión de recordatorios, emergencias y monitoreo de salud. Utiliza modelos locales de ML y opcionalmente OpenAI para generación y validación.

## 2. Arquitectura General
- **FastAPI** como framework principal.
- **SQLModel/SQLAlchemy** para persistencia en base de datos SQLite.
- **Pydantic** para validación de datos y esquemas.
- **Modelos ML locales** (transformers, spaCy) y opcionalmente OpenAI.
- Estructura modular: adapters, core, repositories, schemas, services, api.

## 3. Componentes Principales

### 3.1 Configuración y Entorno
- Archivo: backend/core/config.py
- Clase `Settings` gestiona variables de entorno, rutas de modelos, claves API, parámetros de flujo y decodificador.
- Permite cambiar entre modelos locales y OpenAI.

### 3.2 Modelos de Base de Datos
- Archivo: backend/repositories/db_models.py
- Tablas: User, Device, Session, Message, Reminder, EmergencyEvent.
- Uso de campos JSON para condiciones médicas, análisis ML, metadatos de flujo.

### 3.3 Repositorio y Persistencia
- Archivo: backend/repositories/repository.py
- Funciones CRUD para usuarios, dispositivos, mensajes, recordatorios, emergencias.
- Lógica para evitar duplicados, actualizar estados y vincular entidades.
- Helpers para inicializar y migrar la base de datos.

### 3.4 Adaptadores y Predictores ML
- Archivo: backend/adapters/predictors.py
- Carga y gestión de modelos locales (intención, sentimiento, emoción, NER).
- Funciones para predecir intención, sentimiento, emoción, entidades.
- Safety gate para filtrar mensajes peligrosos.
- Generación de respuestas con OpenAI o mock local.

### 3.5 Orquestador de Conversaciones
- Archivo: backend/services/orchestrator.py
- Clase principal: `ConversationOrchestrator`.
- Decide el flujo de la conversación (emergencia, recordatorio, acompañamiento, consulta).
- Combina predicción local y validación LLM (si está habilitado).
- Persiste eventos y recordatorios según el flujo detectado.

### 3.6 API REST (endpoints)
- Carpeta: backend/api/
- Endpoints para chat, dispositivos, emergencias, salud, recordatorios, usuarios, historial.
- Validación estricta de parámetros y respuestas.
- Uso de dependencias para inyectar el orquestador.

### 3.7 Extracción de Recordatorios
- Archivo: backend/services/reminder_extractor.py
- Heurísticas y regex para extraer título y fecha/hora de recordatorios desde texto libre.
- Uso de spaCy NER y dateparser.

## 4. Flujos Críticos

### 4.1 Conversación y Decisión de Flujos
- El endpoint `/chat` recibe texto, normaliza y persiste el mensaje.
- El orquestador decide el flujo usando predictores ML y opcionalmente LLM.
- Responde con texto, análisis ML, y metadatos para el frontend.

### 4.2 Emergencias
- Detección por intención, safety gate o palabras clave.
- Persistencia y actualización de eventos de emergencia.
- Endpoints para activar, consultar, actualizar y cancelar emergencias.

### 4.3 Recordatorios
- Detección por intención, palabras clave y confirmación del usuario.
- Extracción de contenido y hora, persistencia y deduplicación.
- Endpoints para crear, listar, actualizar y cancelar recordatorios.

### 4.4 Métricas de Salud
- Endpoints para registrar y consultar métricas de salud (ej. presión, glucosa).
- Validación de pertenencia de dispositivo y usuario.

## 5. Esquemas y Validaciones
- Carpeta: backend/schemas/
- Uso de Pydantic para definir requests/responses.
- Validaciones de campos obligatorios, tipos y formatos.
- Enum para estados y tipos críticos (emergencia, recordatorio).

## 6. Seguridad
- Safety gate para filtrar mensajes peligrosos o de emergencia.
- Validación de parámetros en endpoints críticos.
- Persistencia segura y control de acceso por device_id y user_id.

## 7. Dependencias
- Python >=3.10
- FastAPI, SQLModel, Pydantic, transformers, spaCy, dateparser, dotenv, OpenAI
- Modelos locales en carpeta /models/

## 8. Notas de Extensión y Mantenimiento
- Modularidad para añadir nuevos flujos, modelos ML o endpoints.
- Configuración flexible por variables de entorno.
- Documentar nuevos endpoints y actualizar esquemas en schemas/.
- Mantener actualizados los modelos ML y dependencias.

---

**Última actualización:** 11 de diciembre de 2025
