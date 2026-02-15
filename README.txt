================================================================================
 SISTEMA RAG - Evaluación de Preguntas sobre el Paper REFRAG
 Modelización de Empresa (UCM) - Manuel Pontón Sarrió
================================================================================

DESCRIPCIÓN
-----------
Este proyecto implementa un pipeline RAG (Retrieval-Augmented Generation) que
responde preguntas de opción múltiple (A/B/C/D) sobre el artículo académico
"REFRAG: Reducing Computation in Retrieval-Augmented Generation".

El sistema extrae texto del PDF, lo divide en fragmentos semánticos, genera
embeddings vectoriales, y utiliza Google Gemini para seleccionar la respuesta
correcta basándose en los fragmentos más relevantes recuperados por similitud.


CÓMO FUNCIONA (PASO A PASO)
----------------------------

1. CHUNKING (chunking_pipeline.py)
   - Lee el PDF (paper.pdf) y extrae el texto limpio
   - Divide el texto en fragmentos (chunks) usando similitud semántica
   - Genera un embedding de 384 dimensiones para cada chunk
   - Guarda todo en chunks_with_embeddings.json (468 chunks)

2. RETRIEVAL (retriever.py)
   - Para cada pregunta, genera su embedding
   - Calcula la similitud coseno con todos los chunks del PDF
   - Devuelve los K chunks más relevantes (por defecto K=6)

3. EVALUACIÓN (rag_evaluator.py)
   - Carga las 70 preguntas de ModelizaciónEmpresaUCMData.json
   - Para cada pregunta:
     a) Recupera los 6 chunks más relevantes del PDF
     b) Construye un prompt con el contexto + pregunta + opciones
     c) Llama a Google Gemini (gemini-2.5-flash-lite, temperature=0)
     d) Extrae la letra (A/B/C/D) de la respuesta
     e) Compara con la respuesta correcta
   - Muestra el accuracy final y guarda los resultados en JSON


REQUISITOS
----------
- Python 3.10+
- API key de Google Gemini (https://aistudio.google.com/apikey)
- Instalar dependencias:

    pip install -r requirements.txt


USO RÁPIDO
----------

  # 1. Activar el entorno virtual
  source .venv/bin/activate

  # 2. Generar chunks y embeddings (solo la primera vez, tarda ~2 min)
  python chunking_pipeline.py

  # 3. Ejecutar evaluación completa (70 preguntas)
  python rag_evaluator.py --api-key TU_API_KEY

  # 4. Evaluación rápida (21 preguntas, para pruebas)
  python rag_evaluator.py --api-key TU_API_KEY --num-questions 21

  # 5. También se puede definir la API key como variable de entorno
  export GOOGLE_API_KEY=TU_API_KEY
  python rag_evaluator.py


PARÁMETROS DEL EVALUADOR
-------------------------
  --api-key TEXT          API key de Google Gemini (o usar GOOGLE_API_KEY)
  --num-questions N       Número de preguntas a evaluar (default: todas)
  --context-size N        Caracteres máximos por chunk (default: 300)
  --top-k N              Chunks recuperados por pregunta (default: 6)
  --model TEXT            Modelo Gemini (default: gemini-2.5-flash-lite)


ARCHIVOS DEL PROYECTO
---------------------
  chunking_pipeline.py          Pipeline de extracción, chunking y embeddings
  retriever.py                  Recuperación de chunks por similitud coseno
  rag_evaluator.py              Evaluador RAG con Google Gemini
  chunks_with_embeddings.json   468 chunks con embeddings (generado)
  ModelizaciónEmpresaUCMData.json  70 preguntas de evaluación
  paper.pdf                     Artículo académico REFRAG
  requirements.txt              Dependencias Python
  INFORME_RAG_EVALUACION.md     Informe completo de resultados
  Informe.pdf                   Informe del proyecto sin asistencia de IA


RESULTADO OBTENIDO
------------------
  Configuración óptima: context_size=300, top_k=6, gemini-2.5-flash-lite
  Accuracy: 82.86% (58/70 preguntas correctas)
