from typing import Dict, Any
from langchain_openai import ChatOpenAI 
import json


def evaluate_answers(question: str, pipeline_answers:Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    answers_text = ""
    for name, data in pipeline_answers.items():
        answers_text += f"""
        PIPELINE:{name}
        ANSWERS:{data['answer']}

        CONTEXT USED:{(data['contexts'])}
        """
        

    prompt = f"""
       You are a STRICT RAG EVALUATOR.

       You will be given:
       1. A user question
       2. Answers from multiple RAG pipelines
       3. The context used by each pipeline

       RULES:
       - Judge answers ONLY using the provided context
       - Do NOT use your own knowledge
       - Penalize hallucinations
       - Reward factual accuracy, relevance, and completeness

       TASK:
       Score each pipeline from 0 to 10.
       Pick the BEST pipeline.
       Explain briefly why.

       OUTPUT STRICTLY IN JSON FORMAT:

      {{
           "scores": {{
           "pipeline_A": number,
           "pipeline_B": number,
           "pipeline_C": number,
           "pipeline_D": number
            }},
         "best_pipeline": "pipeline_name",
         "reason": "short explanation"
        }}

      QUESTION:
      {question}

      {answers_text}
       """

    response = llm.invoke(prompt)

    

    import json
    import re

    raw = response.content.strip()

# Try direct JSON parse first
    try:
      parsed = json.loads(raw)
    except json.JSONDecodeError:
    # Try to extract JSON from text
      match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "error": "Evaluator returned no JSON",
            "raw_output": raw
        }

    try:
        parsed = json.loads(match.group())
    except Exception:
        return {
            "error": "Evaluator JSON could not be parsed",
            "raw_output": raw
        }

# Validate schema
    required = {"scores", "best_pipeline", "reason"}
    if not required.issubset(parsed.keys()):
      return {
        "error": "Evaluator JSON missing required keys",
        "raw_output": parsed
    }

    return parsed






         




