"""The judge prompts, copied verbatim from the two benchmarks' own evaluation code (both MIT licensed).

``LONGMEMEVAL_*``: LongMemEval's answer-check prompts, one per question type and one for abstention questions, from
``get_anscheck_prompt`` in ``src/evaluation/evaluate_qa.py`` of github.com/xiaowu0162/LongMemEval at commit
``9e0b455f4ef0e2ab8f2e582289761153549043fc`` (``datasets/longmemeval_s.py`` pins that file by hash as
``JUDGE_PIN``). Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory", ICLR
2025. The templates take the question, the correct answer (for a preference question, its rubric; for an abstention
question, the explanation) and the model's response, in that order, through ``str.format``.

``BEAM_RUBRIC``: BEAM's unified judge prompt (``unified_llm_judge_base_prompt`` in ``src/prompts.py`` of
github.com/mohammadtavakoli78/BEAM at commit ``b2da22eac88bb0874c64665f13457eb99835774a``), which its evaluation
(``src/evaluation/compute_metrics.py``) sends once per rubric item with ``<rubric_item>`` and ``<llm_response>``
replaced, and whose ``score`` it averages over the items. Tavakoli et al., "Beyond a Million Tokens: Benchmarking
and Enhancing Long-Term Memory in LLMs", arXiv 2510.27246.

The two licence notices follow, as the licences require of a copy.

LongMemEval (the LONGMEMEVAL_* prompts):

MIT License

Copyright (c) 2024 Di Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

BEAM (the BEAM_RUBRIC prompt):

MIT License

Copyright (c) 2025 Mohammad Tavakoli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

LONGMEMEVAL_ANSWER = (
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
    'response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct '
    'answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the '
    'response only contains a subset of the information required by the answer, answer no. \n'
    '\n'
    'Question: {}\n'
    '\n'
    'Correct Answer: {}\n'
    '\n'
    'Model Response: {}\n'
    '\n'
    'Is the model response correct? Answer yes or no only.'
)

LONGMEMEVAL_TEMPORAL = (
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
    'response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct '
    'answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the '
    'response only contains a subset of the information required by the answer, answer no. In addition, do not '
    'penalize off-by-one errors for the number of days. If the question asks for the number of '
    'days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is '
    "18), the model's response is still correct. \n"
    '\n'
    'Question: {}\n'
    '\n'
    'Correct Answer: {}\n'
    '\n'
    'Model Response: {}\n'
    '\n'
    'Is the model response correct? Answer yes or no only.'
)

LONGMEMEVAL_UPDATE = (
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
    'response contains the correct answer. Otherwise, answer no. If the response contains some previous '
    'information along with an updated answer, the response should be considered as correct as long as the '
    'updated answer is the required answer.\n'
    '\n'
    'Question: {}\n'
    '\n'
    'Correct Answer: {}\n'
    '\n'
    'Model Response: {}\n'
    '\n'
    'Is the model response correct? Answer yes or no only.'
)

LONGMEMEVAL_PREFERENCE = (
    'I will give you a question, a rubric for desired personalized response, and a response from a model. Please '
    'answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to '
    "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
    'personal information correctly.\n'
    '\n'
    'Question: {}\n'
    '\n'
    'Rubric: {}\n'
    '\n'
    'Model Response: {}\n'
    '\n'
    'Is the model response correct? Answer yes or no only.'
)

LONGMEMEVAL_ABSTENTION = (
    'I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if '
    'the model correctly identifies the question as unanswerable. The model could say that the information is '
    'incomplete, or some other information is given but the asked information is not.\n'
    '\n'
    'Question: {}\n'
    '\n'
    'Explanation: {}\n'
    '\n'
    'Model Response: {}\n'
    '\n'
    'Does the model correctly identify the question as unanswerable? Answer yes or no only.'
)

# The question types each template serves (``get_anscheck_prompt``'s branches).
LONGMEMEVAL_BY_TYPE = {
    'single-session-user': LONGMEMEVAL_ANSWER,
    'single-session-assistant': LONGMEMEVAL_ANSWER,
    'multi-session': LONGMEMEVAL_ANSWER,
    'temporal-reasoning': LONGMEMEVAL_TEMPORAL,
    'knowledge-update': LONGMEMEVAL_UPDATE,
    'single-session-preference': LONGMEMEVAL_PREFERENCE,
}

BEAM_RUBRIC = (
    '\n'
    "You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the "
    'specified RUBRIC CRITERION.\n'
    '\n'
    '## EVALUATION INPUTS\n'
    '- RUBRIC CRITERION (what to check): <rubric_item>\n'
    '- RESPONSE TO EVALUATE: <llm_response>\n'
    '\n'
    '## EVALUATION RUBRIC:\n'
    'The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should '
    'demonstrate. \n'
    '\n'
    '**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n'
    '- **Positive requirements** (things the response SHOULD include/do)\n'
    '- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", '
    '"avoid", "absent")\n'
    '\n'
    '## RESPONSIVENESS REQUIREMENT\n'
    'A compliant response must be **on-topic** and attempt to answer it.\n'
    '- If the response does not address the QUESTION, score **0.0** and stop.\n'
    '- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the '
    'prohibited element is absent.\n'
    '\n'
    '## SEMANTIC TOLERANCE RULES:\n'
    'Judge by meaning, not exact wording.\n'
    '- Accept **paraphrases** and **synonyms** that preserve intent.\n'
    '- **Case/punctuation/whitespace** differences must be ignored.\n'
    '- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or '
    '“sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n'
    '- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) '
    'over string matching.\n'
    '\n'
    '## STYLE NEUTRALITY (prevents style contamination):\n'
    'Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure '
    '(e.g., “itemized list”, “no citations”, “one sentence”).\n'
    '- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n'
    '- Only evaluate format when the rubric **explicitly** mandates it.\n'
    '\n'
    '## SCORING SCALE:\n'
    '- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n'
    '  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n'
    '  - Negative: prohibited element **absent** AND response is **responsive**.\n'
    '  \n'
    '- **0.5 (Partial Compliance)**: Partially complies.\n'
    '  - Positive: element present but minor inaccuracies/incomplete execution.\n'
    '  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n'
    '  \n'
    '- **0.0 (No Compliance)**: Fails to comply.\n'
    '  - Positive: required element missing or incorrect.\n'
    '  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is '
    'absent.\n'
    '\n'
    '## EVALUATION INSTRUCTIONS:\n'
    '1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) '
    'or absent (negative/constraint).\n'
    '\n'
    '2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, '
    'evaluate whether:\n'
    '   - **All elements** must be present for full compliance (1.0)\n'
    '   - **Some elements** present indicates partial compliance (0.5)\n'
    '   - **No elements** present indicates no compliance (0.0)\n'
    '   \n'
    '3. **Check Compliance**: \n'
    '   - For positive requirements: Look for the presence and quality of the required element\n'
    '   - For negative constraints: Look for the absence of the prohibited element\n'
    '\n'
    '4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale '
    'above.\n'
    '\n'
    '5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n'
    '\n'
    '## OUTPUT FORMAT:\n'
    'Return your evaluation in JSON format with two fields:\n'
    '\n'
    '{\n'
    '   "score": [your score: 1.0, 0.5, or 0.0],\n'
    '   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified '
    'the assigned score]"\n'
    '}\n'
    '\n'
    'NOTE: ONLY output the json object, without any explanation before or after that\n'
)
