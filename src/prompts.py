"""
Templates used to compute the relevance of a passage, and to the correctness of an answer.
"""


RELEVANCE_TEMPLATE = """
Determine if a document is RELEVANT or IRRELEVANT for answering a question. A document is RELEVANT if it contains information that directly supports at least one acceptable answer.

RELEVANT examples:
- Q: "where does the story the great gatsby take place" | Answers: ['Long Island of 1922']
  Doc: "The Great Gatsby...follows characters living in West Egg on Long Island in summer of 1922"
  → Contains the exact answer

- Q: "when did korn's follow the leader come out" | Answers: ['August 18, 1998', 'Summer 1998']
  Doc: "Follow the Leader...was released on August 18, 1998"
  → Contains the exact release date

IRRELEVANT examples:
- Q: "where does the story the great gatsby take place" | Answers: ['Long Island of 1922']
  Doc: "While Long Island features prominently in American literature, the socioeconomic dynamics..."
  → Mentions Long Island but not as the story's setting

- Q: "who played bobby byrd in get on up" | Answers: ['Nelsan Ellis']
  Doc: "Critics praised the casting of Bobby Byrd and the chemistry between the main characters...
  → Discusses casting but doesn't name the specific actor

Evaluation steps:
1. Find the specific information needed to match an acceptable answer. Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
2. Check if the document contains this information directly or through clear inference
3. Check for these common errors:
    - The document contains similar keywords/themes but not the actual answer
    - The document contains partial information that would need to be combined with external knowledge
    - The document discusses related topics but doesn't specifically answer the question

Here is a new example. Don't apologize or correct yourself if there was a mistake; we are just trying to evaluate the relevance of the document.
```
Question: {question}
Acceptable answers list: {answers}
Document: {document}
```

Evaluate the document for this new question as one of:
A: RELEVANT
B: IRRELEVANT

Return a JSON object with the following format:
{{
    "motivation": "Brief explanation (max 2 sentences)",
    "grade": "A" or "B"
}}
""".strip()


GRADER_TEMPLATE_ABSTAIN = """
Your job is to grade a predicted answer as CORRECT, INCORRECT, or ABSTAINED against a list of acceptable answers.

## Grading Rules

A predicted answer is CORRECT if:
- It contains the essential information from at least one acceptable answer
- It contains no contradictory information
- When the acceptable answer is a list, at least one item from that list is present

A predicted answer is INCORRECT if:
- It contradicts any acceptable answer
- It contains factually wrong information (even with hedging like "I think" or "possibly")

A predicted answer is ABSTAINED if:
- It explicitly refuses to answer or states uncertainty without providing substantive information
- It contains phrases like "I don't know", "I'm not sure", "I cannot answer", "NO-RESPONSE", "I'm uncertain"

## Examples

**CORRECT answers:**
```
Question: What are Barack Obama's children's names?
Acceptable: ['Malia Obama and Sasha Obama', 'Natasha Marian and Malia Ann']
"Malia and Sasha" (essential info present)
"I believe it's Malia and Sasha Obama" (hedging with correct info)
```

**INCORRECT answers:**
```
Question: What are Barack Obama's children's names?
Acceptable: ['Malia Obama and Sasha Obama']
"Only Malia" (incomplete when both required)
"Malia, Sasha, and Susan" (contains wrong info)
"He has no children" (contradicts acceptable answer)
```

**ABSTAINED answers:**
```
Question: What are Barack Obama's children's names?
"I don't know the names of his children"
"I'm not sure about this information"
"I cannot provide that information"
"I'm uncertain and don't want to guess"
"NO-RESPONSE"
```

## Important Notes
- Ignore capitalization, punctuation, grammar, and order.
- Don't penalize omitted information that's clearly inferred from the question.
- Don't penalize minor typos in names.
- Grade using only the question and acceptable answers provided, even if you know the answer is factually wrong.

---

Here is a new example. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Acceptable answers list: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: ABSTAINED

Return a JSON object with the following format:
{{
 "motivation": "Your concise motivation for the grade here. Use maximum 2 sentences.",
 "grade": "A", "B", or "C"
}}
""".strip()

