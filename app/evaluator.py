from typing import Dict, Any
from langchain_openai import ChatOpenAI 


def evaluate_answers(question: str, pipeline_answers:Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    answers_text = ""
    for name, data in pipeline_answers.items():
        answers_text += f"""
        PIPEINE:{name}
        ANSWERS:{data['answer']}

        CONTEXT USED:{" ".join(data['contexts'])}
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

    return response.content
         




