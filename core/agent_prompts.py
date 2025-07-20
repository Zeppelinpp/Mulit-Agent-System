SYSTEM_PROMPT = """
Based on the context and the user quey, analyze user's needs and plan a task list to solve the user's problem.

# Guide
1. Use sequential thinking to understand the user's problem
2. Consider given context if needed
3. Each task should be atomic and self-contained
4. Task can be categorized into sequential or parallel
5. Return the task list in JSON format, schema: {schema}

# Available agents to assign tasks
{workers}

Just Return the task list in JSON format, do not include any other text.
"""

REVIEW_PROMPT = """
Based on the given task goal and result, review if the task is done or not.

# Guide
1. The standard of success is whether the result can reach the goal in certain degree.
2. Return the review result in JSON format, schema: {schema}

JSON Example:
{{
    "is_success": "success",
    "feedback": "The task is done successfully"
}}
{{
    "is_success": "failure",
    "feedback": "The task is not done successfully due to ..."

}}
"""
