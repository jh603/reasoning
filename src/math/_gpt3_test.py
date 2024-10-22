


from src.utils.model_factory import Model


model = Model('gpt-3.5-turbo')

prompt = f"""
You are a college student taking his most important college math test.

Draft a step by step plan to solve the given math problem. In each step, define the inputs to the problem, the problem statement, and the output.
Ensure that the each step is simple and an short explanation of how to complete it.

The format of the output should look like
Step 1:
    Problem: 
    Inputs:
    Outputs:
    
Step 2:
    ...

Suppose that $ABC_4+200_{10}=ABC_9$, where $A$, $B$, and $C$ are valid digits in base 4 and 9. What is the sum when you add all possible values of $A$, all possible values of $B$, and all possible values of $C$?
"""

prompt = "Suppose that $ABC_4+200_{10}=ABC_9$, where $A$, $B$, and $C$ are valid digits in base 4 and 9. What is the sum when you add all possible values of $A$, all possible values of $B$, and all possible values of $C$?"

prompt = f"""
Solve the math problem step by step using any given inputs. Show lots of work.

Suppose that $ABC_4+200_{10}=ABC_9$, where $A$, $B$, and $C$ are valid digits in base 4 and 9. What is the sum when you add all possible values of $A$, all possible values of $B$, and all possible values of $C$?
"""

# prompt = f"""
# Give a short and concise example on how to use inclusion-exclusion principle for counting. Use the format
# Problem:
# Solution:
# """

resp = model.get_response(prompt)

print(resp)