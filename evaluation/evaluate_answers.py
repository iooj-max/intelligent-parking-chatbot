"""
Answer quality evaluation using RAGAS.

Metrics:
- Faithfulness: Factual accuracy (grounded in context)
- Relevance: Answer addresses the question
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from src.chatbot.graph import graph
from src.chatbot.state import ChatbotState

logger = logging.getLogger(__name__)


class AnswerEvaluator:
    """
    Evaluate answer quality using RAGAS.
    """

    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy]

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, float]:
        """
        Evaluate a single answer.

        Args:
            question: User query
            answer: Chatbot response
            context: Retrieved context used to generate answer

        Returns:
            {
                'faithfulness': float,
                'answer_relevancy': float,
            }
        """
        # Create RAGAS dataset format
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [[context]],  # RAGAS expects list of context strings
        }

        dataset = Dataset.from_dict(data)

        # Run RAGAS evaluation
        results = evaluate(dataset, metrics=self.metrics)

        return {
            'faithfulness': results['faithfulness'],
            'answer_relevancy': results['answer_relevancy'],
        }

    def evaluate_with_chatbot(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        Generate answer using chatbot and evaluate.

        Args:
            question: User query

        Returns:
            {
                'question': str,
                'answer': str,
                'context': str,
                'faithfulness': float,
                'answer_relevancy': float,
            }
        """
        # Invoke chatbot graph
        state: ChatbotState = {
            'messages': [HumanMessage(content=question)],
            'mode': 'info',
            'intent': None,
            'context': None,
            'reservation': {'completed_fields': [], 'validation_errors': {}},
            'error': None,
            'iteration_count': 0,
        }

        result = graph.invoke(state)

        # Extract answer and context
        answer = result['messages'][-1].content
        context = result.get('context', '')

        # Evaluate
        scores = self.evaluate_answer(question, answer, context)

        return {
            'question': question,
            'answer': answer,
            'context': context,
            **scores,
        }

    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate answers on entire test dataset.

        Args:
            test_cases: List of test cases [{'question': str}, ...]

        Returns:
            {
                'num_questions': int,
                'avg_faithfulness': float,
                'avg_answer_relevancy': float,
                'per_question_results': List[Dict],
            }
        """
        per_question_results = []

        for test_case in test_cases:
            question = test_case['question']

            logger.info(f"Evaluating question: {question}")
            result = self.evaluate_with_chatbot(question)
            per_question_results.append(result)

        # Calculate averages
        num_questions = len(per_question_results)
        avg_faithfulness = sum(r['faithfulness'] for r in per_question_results) / num_questions
        avg_relevancy = sum(r['answer_relevancy'] for r in per_question_results) / num_questions

        return {
            'num_questions': num_questions,
            'avg_faithfulness': avg_faithfulness,
            'avg_answer_relevancy': avg_relevancy,
            'per_question_results': per_question_results,
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"answer_quality_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved answer quality results to {output_file}")


def main():
    """CLI entry point for answer quality evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate answer quality")
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('evaluation/datasets/answer_test_cases.json'),
        help='Path to test dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation/results'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load test cases
    with open(args.dataset) as f:
        test_cases = json.load(f)

    # Initialize evaluator
    evaluator = AnswerEvaluator()

    # Run evaluation
    logger.info(f"Evaluating {len(test_cases)} test cases...")
    results = evaluator.evaluate_dataset(test_cases)

    # Print summary
    print("\n=== Answer Quality Results ===")
    print(f"Number of questions: {results['num_questions']}")
    print(f"Avg Faithfulness: {results['avg_faithfulness']:.3f}")
    print(f"Avg Relevancy: {results['avg_answer_relevancy']:.3f}")

    # Save results
    evaluator.save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
