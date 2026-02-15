import json
import fitz  # pymupdf
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


class PDFChunker:
    """Extrae y procesa texto de PDFs académicos con limpieza de pies de página"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.document = fitz.open(pdf_path)
        
    def extract_text_clean(self) -> str:
        """Extrae texto del PDF eliminando pies de página comunes"""
        full_text = []
        
        for page_num in range(len(self.document)):
            page = self.document[page_num]
            text = page.get_text()
            
            # Eliminar pies de página comunes (últimas líneas, números de página)
            lines = text.split('\n')
            
            # Filtrar líneas que parecen pies de página
            filtered_lines = []
            for line in lines:
                # Ignorar líneas vacías y líneas que son solo números
                if not line.strip():
                    continue
                # Ignorar líneas cortas que parecen números de página
                if len(line.strip()) <= 3 and line.strip().isdigit():
                    continue
                # Ignorar URLs o referencias de pie de página muy cortas
                if len(line.strip()) < 10 and any(x in line.lower() for x in ['http', 'www', '©', '®']):
                    continue
                filtered_lines.append(line)
            
            full_text.append('\n'.join(filtered_lines))
        
        return '\n\n'.join(full_text)
    
    def close(self):
        self.document.close()


class SemanticChunker:
    """Dividir texto usando similitud semántica entre oraciones"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones manteniendo estructura"""
        # Usar puntuación común para dividir
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtrar oraciones vacías
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calcula similitud coseno entre dos oraciones"""
        embeddings = self.model.encode([sentence1, sentence2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10
        )
        return float(similarity)
    
    def chunk(self, text: str, max_chunk_size: int = 500, 
              similarity_threshold: float = 0.5) -> List[Dict[str, str]]:
        """
        Chunking semántico: agrupa oraciones similares
        
        Args:
            text: Texto a dividir
            max_chunk_size: Tamaño máximo en caracteres por chunk
            similarity_threshold: Umbral de similitud para agrupar (0-1)
        
        Returns:
            Lista de chunks con metadata
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Si agregando la oracion superamos el límite
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Crear chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'sentences': len(current_chunk),
                    'char_length': len(chunk_text),
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i
                })
                current_chunk = []
                current_size = 0
            
            # Verificar similitud con última oración del chunk
            if current_chunk:
                similarity = self._calculate_similarity(current_chunk[-1], sentence)
                # Si la similitud es baja, podríamos empezar nuevo chunk
                if similarity < similarity_threshold and current_size > 100:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'sentences': len(current_chunk),
                        'char_length': len(chunk_text),
                        'start_sentence': i - len(current_chunk),
                        'end_sentence': i
                    })
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 por espacio
        
        # Agregar último chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'sentences': len(current_chunk),
                'char_length': len(chunk_text),
                'start_sentence': len(sentences) - len(current_chunk),
                'end_sentence': len(sentences)
            })
        
        return chunks


class JSONChunker:
    """Procesa preguntas de múltiple opción del JSON"""
    
    @staticmethod
    def chunk_questions(json_path: str, 
                       max_chunk_size: int = 500) -> List[Dict]:
        """
        Lee JSON y crea chunks para cada pregunta + opciones
        
        Args:
            json_path: Ruta al JSON
            max_chunk_size: Tamaño máximo por chunk
        
        Returns:
            Lista de chunks procesados
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        chunks = []
        for idx, q in enumerate(questions):
            # Combinar pregunta + opciones en un chunk
            question_text = q['question']
            options_text = '\n'.join([
                f"{key}: {value}" 
                for key, value in q['answers'].items()
            ])
            
            full_text = f"{question_text}\n\n{options_text}"
            
            chunk = {
                'text': full_text,
                'source': 'json_question',
                'question_id': idx,
                'question': question_text,
                'answers': q['answers'],
                'correct_answer': q['correct_answer'],
                'paper_reference': q['paper_reference'],
                'char_length': len(full_text)
            }
            chunks.append(chunk)
        
        return chunks


class EmbeddingGenerator:
    """Genera embeddings para chunks usando sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Genera embeddings para cada chunk
        
        Args:
            chunks: Lista de chunks con campo 'text'
        
        Returns:
            Chunks con embeddings agregados
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks


def main():
    """Pipeline completo de chunking"""
    
    pdf_path = "paper.pdf"
    json_path = "ModelizaciónEmpresaUCMData.json"
    
    print("=" * 60)
    print("INICIANDO PIPELINE DE CHUNKING SEMÁNTICO")
    print("=" * 60)
    
    # 1. Procesar PDF
    print("\n[1/4] Extrayendo texto del PDF...")
    pdf_chunker = PDFChunker(pdf_path)
    pdf_text = pdf_chunker.extract_text_clean()
    pdf_chunker.close()
    print(f"✓ Texto extraído: {len(pdf_text)} caracteres")
    
    # 2. Aplicar semantic chunking al PDF
    print("\n[2/4] Aplicando semantic chunking al PDF...")
    semantic_chunker = SemanticChunker()
    pdf_chunks = semantic_chunker.chunk(pdf_text, max_chunk_size=500, similarity_threshold=0.5)
    
    # Agregar metadata de origen
    for i, chunk in enumerate(pdf_chunks):
        chunk['source'] = 'pdf'
        chunk['chunk_id'] = i
    
    print(f"✓ Chunks del PDF: {len(pdf_chunks)}")
    print(f"  - Tamaño promedio: {np.mean([c['char_length'] for c in pdf_chunks]):.0f} caracteres")
    print(f"  - Rango: {min(c['char_length'] for c in pdf_chunks)} - {max(c['char_length'] for c in pdf_chunks)} caracteres")
    
    # 3. Generar embeddings SOLO para PDF chunks
    print("\n[3/4] Generando embeddings...")
    embedding_generator = EmbeddingGenerator()
    chunks_with_embeddings = embedding_generator.generate_embeddings(pdf_chunks)
    
    print(f"✓ Total de chunks con embeddings: {len(chunks_with_embeddings)}")
    
    # 4. Guardar resultados
    output_file = "chunks_with_embeddings.json"
    
    output_data = {
        'chunks': chunks_with_embeddings,
        'metadata': {
            'total_chunks': len(chunks_with_embeddings),
            'pdf_chunks': len(pdf_chunks),
            'model': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
            'embedding_dimension': 384
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Resultados guardados en: {output_file}")
    
    # Mostrar ejemplo
    print("\n" + "=" * 60)
    print("EJEMPLO DE CHUNK GENERADO")
    print("=" * 60)
    
    print("\n[Chunk del PDF]")
    pdf_example = pdf_chunks[0]
    print(f"Fuente: {pdf_example['source']}")
    print(f"Caracteres: {pdf_example['char_length']}")
    print(f"Oraciones: {pdf_example['sentences']}")
    print(f"Texto: {pdf_example['text'][:200]}...")


if __name__ == "__main__":
    main()
