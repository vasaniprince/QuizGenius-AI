import os
import asyncio
import json
import logging
from typing import Dict, List, Any

# AI and Data Processing Libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QuizDataProcessor:
    @classmethod
    def process_comprehensive_analysis(
        cls, 
        current_quiz: Dict, 
        quiz_submission: Dict, 
        historical_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Comprehensive quiz performance analysis
        
        Args:
            current_quiz (Dict): Current quiz details
            quiz_submission (Dict): Submission details
            historical_data (List[Dict]): Historical quiz performances
        
        Returns:
            Detailed performance analysis
        """
        try:
            # Overall Performance Metrics
            performance_metrics = {
                'current_quiz': {
                    'title': current_quiz.get('quiz', {}).get('title', 'Unknown Quiz'),
                    'total_questions': quiz_submission.get('total_questions', 0),
                    'correct_answers': quiz_submission.get('correct_answers', 0),
                    'incorrect_answers': quiz_submission.get('incorrect_answers', 0),
                    'accuracy': quiz_submission.get('accuracy', '0%'),
                    'rank_text': quiz_submission.get('rank_text', 'N/A')
                },
                'topic_analysis': cls._analyze_topic_performance(
                    current_quiz.get('questions', []), 
                    quiz_submission.get('response_map', {})
                ),
                'response_patterns': {
                    'time_taken': quiz_submission.get('duration', '0'),
                    'mistakes_corrected': quiz_submission.get('mistakes_corrected', 0),
                    'initial_mistake_count': quiz_submission.get('initial_mistake_count', 0)
                },
                'historical_performance': cls._analyze_historical_performance(historical_data)
            }
            
            return performance_metrics
        
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            return {}
    
    @staticmethod
    def _analyze_topic_performance(
        questions: List[Dict], 
        response_map: Dict
    ) -> Dict[str, Dict]:
        """
        Detailed topic-wise performance analysis
        
        Args:
            questions (List[Dict]): Quiz questions
            response_map (Dict): User's response mappings
        
        Returns:
            Topic performance breakdown
        """
        topic_performance = {}
        
        for question in questions:
            question_id = str(question.get('id', ''))
            topic = question.get('topic', 'Uncategorized').strip()
            
            if topic not in topic_performance:
                topic_performance[topic] = {
                    'total_questions': 0,
                    'attempted_questions': 0,
                    'correct_questions': 0,
                    'difficulty_distribution': {}
                }
            
            topic_performance[topic]['total_questions'] += 1
            
            # Check if question was attempted
            if question_id in response_map:
                topic_performance[topic]['attempted_questions'] += 1
                
                # Check correctness
                selected_option_id = response_map[question_id]
                correct_options = [
                    opt['id'] for opt in question.get('options', []) 
                    if opt.get('is_correct', False)
                ]
                
                if selected_option_id in correct_options:
                    topic_performance[topic]['correct_questions'] += 1
                
                # Difficulty tracking
                difficulty = question.get('difficulty_level', 'medium')
                topic_performance[topic]['difficulty_distribution'][difficulty] = \
                    topic_performance[topic]['difficulty_distribution'].get(difficulty, 0) + 1
        
        # Calculate accuracies
        for topic, stats in topic_performance.items():
            stats['accuracy'] = (stats['correct_questions'] / stats['total_questions'] * 100) \
                if stats['total_questions'] > 0 else 0
        
        return topic_performance
    
    @staticmethod
    def _analyze_historical_performance(historical_data: List[Dict]) -> Dict:
        """
        Analyze performance across historical quizzes
        
        Args:
            historical_data (List[Dict]): Previous quiz performances
        
        Returns:
            Historical performance insights
        """
        if not historical_data:
            return {}
        
        performances = []
        topics = set()
        
        for quiz in historical_data:
            performances.append({
                'score': quiz.get('score', 0),
                'accuracy': float(quiz.get('accuracy', '0%').strip('%')),
                'topic': quiz.get('quiz', {}).get('topic', 'Unknown')
            })
            topics.add(quiz.get('quiz', {}).get('topic', 'Unknown'))
        
        return {
            'average_score': np.mean([p['score'] for p in performances]) if performances else 0,
            'average_accuracy': np.mean([p['accuracy'] for p in performances]) if performances else 0,
            'unique_topics': list(topics)
        }

class NeetTestlineRecommendationSystem:
    def __init__(self, groq_api_key: str):
        """
        Initialize recommendation system
        
        Args:
            groq_api_key (str): Groq API key for LLM
        """
        self.data_processor = QuizDataProcessor()
        
        # Initialize LLM for recommendations
        self.llm = ChatGroq(
            temperature=0.2, 
            model_name="llama3-70b-8192",
            groq_api_key=groq_api_key
        )
    
    async def generate_personalized_recommendations(
        self, 
        current_quiz: Dict, 
        quiz_submission: Dict, 
        historical_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive personalized recommendations
        
        Args:
            current_quiz (Dict): Current quiz details
            quiz_submission (Dict): Submission details
            historical_data (List[Dict]): Historical quiz performances
        
        Returns:
            Personalized recommendation report
        """
        try:
            # Process comprehensive quiz data
            performance_metrics = self.data_processor.process_comprehensive_analysis(
                current_quiz, 
                quiz_submission, 
                historical_data
            )
            
            # Generate AI-powered recommendations
            recommendations = await self._generate_ai_recommendations(performance_metrics)
            
            return {
                'performance_metrics': performance_metrics,
                'recommendations': recommendations
            }
        
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return None
    
    async def _generate_ai_recommendations(self, performance_metrics: Dict):
        """
        Generate personalized recommendations using LLM
        
        Args:
            performance_metrics (Dict): Processed performance data
        
        Returns:
            AI-generated recommendations
        """
        recommendation_prompt = ChatPromptTemplate.from_template("""
        Analyze the student's performance metrics and provide personalized study recommendations:

        Current Quiz Performance:
        {current_quiz_details}

        Topic Analysis:
        {topic_performance}

        Historical Performance:
        {historical_performance}

        Generate a comprehensive study strategy addressing:
        1. Key improvement areas
        2. Targeted learning recommendations
        3. Strategies for weak topics
        4. Motivation and progress tracking
        """)
        
        # Prepare prompt inputs
        prompt_inputs = {
            'current_quiz_details': json.dumps(performance_metrics.get('current_quiz', {}), indent=2),
            'topic_performance': json.dumps(performance_metrics.get('topic_analysis', {}), indent=2),
            'historical_performance': json.dumps(performance_metrics.get('historical_performance', {}), indent=2)
        }
        
        # Generate recommendations
        chain = recommendation_prompt | self.llm
        recommendations = chain.invoke(prompt_inputs)
        
        return recommendations.content

async def main():
    """Main execution function for demonstration"""
    # Load sample data from files
    with open('Current_Quiz.json', 'r') as f:
        current_quiz = json.load(f)
    
    with open('Quiz_Submission.json', 'r') as f:
        quiz_submission = json.load(f)
    
    with open('History.json', 'r') as f:
        historical_data = json.load(f)
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    recommendation_system = NeetTestlineRecommendationSystem(
        groq_api_key=GROQ_API_KEY
    )
    
    result = await recommendation_system.generate_personalized_recommendations(
        current_quiz, 
        quiz_submission, 
        historical_data
    )
    
    if result:
        # Pretty print performance metrics and recommendations
        print("Performance Metrics:")
        print(json.dumps(result['performance_metrics'], indent=2))
        
        print("\nRecommendations:")
        print(result['recommendations'])
    else:
        print("Failed to generate recommendations")

if __name__ == "__main__":
    asyncio.run(main())