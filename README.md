# QuizGenius AI

## Project Overview
This Python-based application provides personalized learning recommendations for medical students by analyzing quiz performance using advanced data processing and AI-driven insights.

## Key Features
- Comprehensive quiz performance analysis
- Topic-wise performance breakdown
- Historical performance tracking
- AI-generated personalized study recommendations
- Detailed insights using Langchain and Groq LLM

## Technical Stack
- Python
- Langchain
- Groq AI (Llama3 70B model)
- Numpy for statistical analysis
- Async programming with asyncio

## Prerequisites
- Python 3.8+
- Groq API Key
- Required Python packages (see `requirements.txt`)

## Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/vasaniprince/QuizGenius-AI.git
cd your_File_Path
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set Groq API Key
```bash
export GROQ_API_KEY='your_groq_api_key_here'
```

## Usage
```bash
python Your_app_name.py
```

## Core Components
### 1. QuizDataProcessor
- Analyzes current and historical quiz performances
- Generates detailed performance metrics
- Provides topic-wise performance insights

### 2. NeetTestlineRecommendationSystem
- Generates personalized study recommendations
- Uses AI to create targeted learning strategies
- Processes performance data comprehensively

## Sample Input Files
- `Current_Quiz.json`: Current quiz details
- `Quiz_Submission.json`: User's quiz submission
- `History.json`: Historical quiz performances

## Performance Analysis Metrics
- Accuracy tracking
- Topic-level performance
- Difficulty level insights
- Historical performance trends

## AI Recommendation Focus
- Identify key improvement areas
- Provide targeted learning recommendations
- Develop strategies for weak topics
- Motivational progress tracking

## Logging
- Detailed logging for error tracking
- Configurable log levels

## Future Enhancements
- Integration with learning management systems
- More granular recommendation algorithms
- Enhanced visualization of performance metrics

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact
   vasaniprince640@gmail.com