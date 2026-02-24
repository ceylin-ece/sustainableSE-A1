import re
text = """Here is the code:
```python
def example(s):
    return s + "```"
```
And some text after.
"""

def extract_code(text):
    match = re.search(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    code = match.group(1).strip() if match else text.strip()
    return code

print(repr(extract_code(text)))

# Test Phi failure case
phi_fail_text = """```python
def remove_char(s, c):
    # This might have failed because the stop token triggered mid-string
    return s.replace(c, '')
```
"""
print(repr(extract_code(phi_fail_text)))
