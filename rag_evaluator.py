import argparse
import json
import re
import os
from typing import List, Dict
from google.genai import Client
from retriever import RetrieverRAG, load_json_questions


class RAGEvaluator:
    """Evalúa performance del sistema RAG usando Google Gemini (google-genai)"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", context_size: int = 300):
        """
        Inicializa el evaluador con Google Gemini
        
        Args:
            api_key: API key de Google (REQUERIDA)
            model: Modelo a usar (default: gemini-2.5-flash-lite)
            context_size: Caracteres máximos por chunk (default: 300)
        """
        self.client = Client(api_key=api_key)
        self.retriever = RetrieverRAG()
        self.model = model
        self.context_size = context_size
    
    def extract_answer_letter(self, response_text: str) -> str:
        """
        Extrae SOLO la letra (A, B, C, D) de la respuesta del modelo
        
        Args:
            response_text: Texto completo de la respuesta
            
        Returns:
            Letra única (A, B, C o D) o "INVALID" si no encuentra
        """
        # Validar que response_text no sea None
        if response_text is None or response_text == "":
            return "INVALID"
        
        # Limpiar: remove backticks, spaces, etc
        text = response_text.strip().replace('`', '').replace('"', '').replace("'", '').upper()
        
        # Si es una sola letra válida
        if text in ['A', 'B', 'C', 'D']:
            return text
        
        # Buscar primer [A-D] en la respuesta
        match = re.search(r'[A-D]', text)
        if match:
            return match.group()
        
        return "INVALID"
    
    def build_prompt(self, question: str, answers: Dict, context_chunks: List[str]) -> str:
        """
        Construye un prompt corto que FUERZA respuesta simple
        
        Args:
            question: Pregunta
            answers: Dict con opciones A, B, C, D
            context_chunks: Lista de chunks relevantes del PDF
            
        Returns:
            Prompt formateado
        """
        # Truncar contexto a context_size chars por chunk
        context_text = "\n".join([chunk[:self.context_size] for chunk in context_chunks])
        
        prompt = f"""Contexto del documento:
{context_text}

Pregunta: {question}

Opciones:
A) {answers['A'][:100]}
B) {answers['B'][:100]}
C) {answers['C'][:100]}
D) {answers['D'][:100]}

Based on the context provided, select ONLY the correct letter (A, B, C, or D):"""
        
        return prompt
    
    def evaluate_question(self, question_data: Dict, question_id: int, top_k: int = 6) -> Dict:
        """
        Evalúa una pregunta con el sistema RAG
        
        Args:
            question_data: Datos de la pregunta del JSON
            question_id: ID de la pregunta
            top_k: Número de chunks a recuperar
            
        Returns:
            Dict con resultados de la evaluación
        """
        question = question_data['question']
        answers = question_data['answers']
        correct_answer = question_data['correct_answer']
        
        # 1. Recuperar chunks relevantes del PDF
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        context_chunks = [chunk['text'] for chunk in retrieved_chunks]
        similarities = [chunk['similarity_score'] for chunk in retrieved_chunks]
        
        # 2. Construir prompt
        prompt = self.build_prompt(question, answers, context_chunks)
        
        # 3. Llamar a Google Gemini
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 10
                }
            )
            raw_response = response.text if response and response.text else ""
        except Exception as e:
            print(f"[ERROR en pregunta {question_id}] {str(e)}")
            return {
                'question_id': question_id,
                'question': question[:100] + "..." if len(question) > 100 else question,
                'correct_answer': correct_answer,
                'model_answer': 'ERROR',
                'is_correct': False,
                'error': str(e),
                'retrieved_chunks_similarity': [round(s, 3) for s in similarities],
                'raw_response': str(e)[:150]
            }
        
        # 4. Extraer letra de la respuesta
        model_answer = self.extract_answer_letter(raw_response)
        
        # Debug: mostrar respuesta cruda si es INVALID
        if model_answer == "INVALID":
            print(f"    [DEBUG] Respuesta cruda: '{raw_response}'")
        
        # 5. Comparar con respuesta correcta
        is_correct = model_answer == correct_answer
        
        return {
            'question_id': question_id,
            'question': question[:100] + "..." if len(question) > 100 else question,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'raw_response': raw_response[:150],
            'retrieved_chunks_similarity': [round(s, 3) for s in similarities],
            'min_similarity': round(min(similarities), 3) if similarities else 0.0,
            'max_similarity': round(max(similarities), 3) if similarities else 0.0
        }
    
    def evaluate_all(self, num_questions: int = None, top_k: int = 6) -> Dict:
        """
        Evalúa todas las preguntas del JSON
        
        Args:
            num_questions: Número de preguntas a evaluar (None = todas)
            top_k: Número de chunks a recuperar por pregunta
            
        Returns:
            Dict con resultados agregados
        """
        # Cargar preguntas
        questions = load_json_questions()
        
        if num_questions:
            questions = questions[:num_questions]
        
        results = []
        correct_count = 0
        
        print("=" * 70)
        print(f"EVALUANDO {len(questions)} PREGUNTAS")
        print(f"Modelo: {self.model} | Context Size: {self.context_size} | Top-K: {top_k}")
        print("=" * 70)
        
        for idx, q_data in enumerate(questions, 1):
            print(f"[{idx:3d}/{len(questions)}] Evaluando pregunta {idx}...", end=" ", flush=True)
            
            result = self.evaluate_question(q_data, idx, top_k=top_k)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
                print("✓")
            else:
                print(f"✗ (modelo: {result['model_answer']}, correcta: {result['correct_answer']})")  
        
        # Calcular métricas
        accuracy = correct_count / len(questions) if questions else 0
        
        summary = {
            'model': self.model,
            'context_size': self.context_size,
            'top_k': top_k,
            'total_questions': len(questions),
            'correct_answers': correct_count,
            'incorrect_answers': len(questions) - correct_count,
            'accuracy': accuracy,
            'accuracy_percentage': f"{accuracy * 100:.2f}%",
            'results': results
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Imprime un resumen de los resultados"""
        print("\n" + "=" * 70)
        print("RESUMEN DE EVALUACIÓN RAG")
        print("=" * 70)
        print(f"Modelo: {summary.get('model', 'N/A')}")
        print(f"Context Size: {summary.get('context_size', 'N/A')} chars")
        print(f"Top-K: {summary.get('top_k', 'N/A')}")
        print(f"Total de preguntas: {summary['total_questions']}")
        print(f"Respuestas correctas: {summary['correct_answers']}")
        print(f"Respuestas incorrectas: {summary['incorrect_answers']}")
        print(f"Accuracy: {summary['accuracy_percentage']}")
        print("=" * 70)
        
        # Análisis de similitudes
        similarities_correctas = []
        similarities_incorrectas = []
        for result in summary['results']:
            if result['is_correct']:
                similarities_correctas.extend(result['retrieved_chunks_similarity'])
            else:
                similarities_incorrectas.extend(result['retrieved_chunks_similarity'])
        
        if similarities_correctas:
            print(f"Promedio similitud (respuestas CORRECTAS): {sum(similarities_correctas)/len(similarities_correctas):.3f}")
        if similarities_incorrectas:
            print(f"Promedio similitud (respuestas INCORRECTAS): {sum(similarities_incorrectas)/len(similarities_incorrectas):.3f}")
        print()
        
        # Mostrar algunos ejemplos
        print("EJEMPLOS DE RESULTADOS (primeras 5):")
        print()
        
        for result in summary['results'][:5]:
            status = "✓" if result['is_correct'] else "✗"
            print(f"{status} Pregunta {result['question_id']}: {result['question']}")
            print(f"  Respuesta correcta: {result['correct_answer']}")
            print(f"  Respuesta del modelo: {result['model_answer']}")
            print(f"  Similitudes: {result['retrieved_chunks_similarity']} (min: {result['min_similarity']:.3f}, max: {result['max_similarity']:.3f})")
            print()


def main():
    """Script principal de evaluación (acepta --api-key, --num-questions, --model, --context-size, --top-k)."""

    parser = argparse.ArgumentParser(description="Evaluador RAG mejorado con Google Gemini")
    parser.add_argument('--api-key', help='API key de Google (opcional, también se puede usar GOOGLE_API_KEY env var)')
    parser.add_argument('--num-questions', type=int, default=None, help='Número de preguntas a evaluar (None = todas)')
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='Modelo Gemini a usar (default: gemini-2.5-flash-lite)')
    parser.add_argument('--context-size', type=int, default=300, help='Caracteres máximos por chunk (default: 300)')
    parser.add_argument('--top-k', type=int, default=6, help='Número de chunks a recuperar (default: 6)')
    args = parser.parse_args()

    print("=" * 70)
    print("EVALUADOR RAG MEJORADO CON GOOGLE GEMINI")
    print("=" * 70)

    # Obtener API key desde CLI -> ENV -> prompt
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        api_key = input("\nIntroduce tu API key de Google Gemini: ").strip()

    if not api_key:
        print("❌ API key requerida")
        return

    print(f"\nInicializando evaluador RAG...")
    print(f"  Modelo: {args.model}")
    print(f"  Context Size: {args.context_size} chars")
    print(f"  Top-K: {args.top_k}")
    
    try:
        evaluator = RAGEvaluator(api_key=api_key, model=args.model, context_size=args.context_size)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("✓ Evaluador inicializado correctamente\n")

    # Determinar número de preguntas
    num_questions = args.num_questions
    if num_questions is None:
        env_n = os.getenv('NUM_QUESTIONS')
        if env_n:
            try:
                num_questions = int(env_n)
            except Exception:
                num_questions = None

    # Evaluar
    summary = evaluator.evaluate_all(num_questions=num_questions, top_k=args.top_k)

    # Mostrar resumen
    evaluator.print_summary(summary)

    # Guardar resultados completos con timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rag_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✓ Resultados guardados en: {output_file}")


if __name__ == "__main__":
    main()
