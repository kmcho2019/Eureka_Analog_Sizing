import re
import ast


def extract_first_python_code_block(markdown_text):
    """
    Extract the first Python code block from a markdown string.
    """
    code_block_pattern = re.compile(r'```python\n(.*?)```', re.DOTALL)
    match = code_block_pattern.search(markdown_text)
    
    if match:
        return match.group(1)
    else:
        return None

def extract_top_level_function_code(python_code):
    """
    Extract the top-level function code from a string of Python code.
    """
    class FunctionExtractor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(node)
            # Do not visit child nodes
            return

    extractor = FunctionExtractor()

    try:
        parsed_code = ast.parse(python_code)
    except SyntaxError:
        return None
    extractor.visit(parsed_code)
    
    if extractor.functions:
        first_function = extractor.functions[0]
        start_lineno = first_function.lineno
        end_lineno = first_function.body[-1].end_lineno if hasattr(first_function.body[-1], 'end_lineno') else first_function.body[-1].lineno
        function_lines = python_code.splitlines()[start_lineno - 1:end_lineno]
        return '\n'.join(function_lines)
    else:
        return None

def parse_markdown_for_first_function_code(markdown_text):
    """
    Parse the markdown to find the first Python code block and extract the top-level function code.
    """
    python_code = extract_first_python_code_block(markdown_text)
    
    if python_code:
        function_code = extract_top_level_function_code(python_code)
        return function_code
    else:
        return None