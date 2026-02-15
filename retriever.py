import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


class RetrieverRAG:
    """Busca chunks relevantes usando embeddings semánticos"""
    
    def __init__(self, chunks_file: str = "chunks_with_embeddings.json",
                 model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        # Cargar chunks con embeddings
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        
        # Convertir embeddings a numpy arrays
        for chunk in self.chunks:
            if isinstance(chunk['embedding'], list):
                chunk['embedding'] = np.array(chunk['embedding'])
        
        print(f"✓ Cargados {len(self.chunks)} chunks")
        print(f"  - Modelo: {self.metadata['model']}")
    
    def retrieve(self, query: str, top_k: int = 5, source_filter: str = None) -> List[Dict]:
        """
        Busca los chunks más relevantes para una query
        
        Args:
            query: Pregunta o búsqueda
            top_k: Número de resultados a retornar
            source_filter: Filtrar por 'pdf' o 'json_question' (None = ambos)
        
        Returns:
            Lista de chunks ordenados por relevancia
        """
        # Generar embedding para la query
        query_embedding = self.model.encode(query)
        query_embedding = np.array(query_embedding)
        
        # Calcular similitud con todos los chunks
        similarities = []
        for i, chunk in enumerate(self.chunks):
            # Filtrar por fuente si se especifica
            if source_filter and chunk.get('source') != source_filter:
                continue
            
            # Similitud coseno
            similarity = np.dot(query_embedding, chunk['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk['embedding']) + 1e-10
            )
            similarities.append((i, float(similarity), chunk))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top_k
        results = []
        for idx, similarity, chunk in similarities[:top_k]:
            chunk_copy = chunk.copy()
            chunk_copy['similarity_score'] = similarity
            chunk_copy['embedding'] = None  # No incluir embedding en resultados para claridad
            results.append(chunk_copy)
        
        return results
    
    def retrieve_from_pdf(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recupera solo del PDF"""
        return self.retrieve(query, top_k=top_k, source_filter='pdf')
    
    def retrieve_from_json(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recupera solo de las preguntas JSON"""
        return self.retrieve(query, top_k=top_k, source_filter='json_question')


def load_json_questions(json_path: str = "ModelizaciónEmpresaUCMData.json") -> List[Dict]:
    """Carga las preguntas del JSON para usar como queries de prueba"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def demo_retrieval():
    """Demostración del sistema de retrieval con preguntas del JSON como queries"""
    
    print("=" * 70)
    print("DEMOSTRACIÓN: RETRIEVAL CON PREGUNTAS DEL JSON")
    print("=" * 70)
    
    # Inicializar retriever
    print("\n[Cargando chunks con embeddings...]")
    retriever = RetrieverRAG()
    
    # Cargar preguntas del JSON para usarlas como queries
    print("[Cargando preguntas del JSON...]")
    questions = load_json_questions()
    print(f"✓ Cargadas {len(questions)} preguntas de prueba\n")
    
    # Probar con las primeras 3 preguntas
    for idx, q_data in enumerate(questions[:3], 1):
        query = q_data['question']
        correct_answer = q_data['correct_answer']
        
        print("\n" + "=" * 70)
        print(f"PREGUNTA {idx}:")
        print(f"{query}")
        print(f"Respuesta correcta en JSON: {correct_answer}")
        print("=" * 70)
        
        # Buscar en PDF chunks basado en esta pregunta
        results = retriever.retrieve(query, top_k=3)
        
        print("\nChunks del PDF más relevantes:\n")
        for i, result in enumerate(results, 1):
            print(f"[Resultado {i}] (Similitud: {result['similarity_score']:.3f})")
            text_preview = result['text'][:200].replace('\n', ' ')
            print(f"Contenido: {text_preview}...\n")
    
    print("\n" + "=" * 70)
    print("FIN DE LA DEMOSTRACIÓN")
    print("=" * 70)


if __name__ == "__main__":
    demo_retrieval()
