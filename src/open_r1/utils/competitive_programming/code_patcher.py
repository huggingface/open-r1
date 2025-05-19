import re

def fix_python3_imports(source_code):
    """
    Fix common import and function changes between Python 3 versions
    
    Args:
        source_code (str): The Python source code to update
        
    Returns:
        str: The updated source code
    """
    # Dictionary of patterns to replacements
    replacements = [
    
        
        # Fix collections.abc imports (changed in Python 3.3+)
        (r'from collections import (Mapping|Sequence|Set|Container|MutableMapping|MutableSet|MutableSequence)', 
         r'from collections.abc import \1'),
        
        # Fix imp module deprecation (deprecated in 3.4)
        (r'import imp', r'import importlib'),
        
        # Fix asyncio.async() to asyncio.ensure_future() (renamed in 3.4.4)
        (r'asyncio\.async\(', r'asyncio.ensure_future('),
        
        # Fix inspect.getargspec to inspect.getfullargspec (deprecated in 3.5)
        (r'inspect\.getargspec', r'inspect.getfullargspec'),
        
        # Fix array.array 'c' type code to 'b' (removed in 3.9)
        (r"array\.array\('c'", r"array.array('b'"),

        # Fix backslash line continuation with multiple newlines (Python-specific issue)
        (r'\\(\r\n|\r|\n)+', '\\\n'),

        # some solutions use getlogin() to check if they are debugging or on an actual submission
        (r'(?:os\s*\.\s*)?getlogin\s*\(\s*\)', 'False'),

        # Fix usage of fractions.gcd (moved to math in 3.5)
        # 1. Fix direct usage: fractions.gcd -> math.gcd
        (r'\bfractions\.gcd\b', r'math.gcd'),

        # 2. Fix 'from fractions import gcd, X' -> 'from fractions import X' (start/middle)
        (r'(from\s+fractions\s+import\s+(?:\([^)]*)?)\bgcd\s*,\s*', r'\1'),

        # 3. Fix 'from fractions import X, gcd' -> 'from fractions import X' (end)
        (r'(from\s+fractions\s+import\s+.*?\S)\s*,\s*\bgcd(\s*\)?\s*(?:#.*)?)', r'\1\2'),

        # 4. Fix standalone 'from fractions import gcd' -> 'from math import gcd'
        (r'from\s+fractions\s+import\s+\(?\s*gcd\s*\)?', r''),
        # --- End: Replacement for the faulty line ---
    ]
    
    lines = source_code.splitlines()
    last_import = max([i for i, line in enumerate(lines) if line.strip().startswith("import") or (line.strip().startswith("from") and "import" in line)], default=0)
    import_section = "\n".join(lines[:last_import + 1])
    main_source = "\n".join(lines[last_import:])
    
    if "fractions.gcd" in source_code and "import math" not in source_code:
        import_section += "\nimport math"
    elif "gcd" in source_code and "from math import gcd" not in source_code:
        import_section += "\nfrom math import gcd"

    if "set_int_max_str_digits" not in source_code:
        import_section += "\nimport sys\nsys.set_int_max_str_digits(0)"

    source_code = import_section + "\n" + main_source

    # Apply each replacement
    for pattern, replacement in replacements:
        source_code = re.sub(pattern, replacement, source_code)

    source_code = source_code.rstrip("\\")
    
    return source_code

def fix_cpp_includes(source_code):
    # has most of the useful functions
    code_header = '#include <bits/stdc++.h>\n'
    # use namespace std since models forget std:: often
    if "using namespace std;" not in source_code and "std::" not in source_code:
        code_header += "\nusing namespace std;\n\n"
    return code_header + source_code

def is_patchable(lang):
    return lang in ("python", "python3", "Python 3", "PyPy 3", "PyPy 3-64", "cpp") or "C++" in lang

def patch_code(text, lang):
    if not text:
        return text
    if lang in ("python", "python3", "Python 3", "PyPy 3", "PyPy 3-64"):
        return fix_python3_imports(text)
    elif "cpp" in lang or "C++" in lang:
        return fix_cpp_includes(text)
    return text


tests = [
    """read = lambda: map(int, input().split())
n, m, z = read()
from fractions import gcd
ans = z // (n * m // gcd(n, m))
print(ans)""",
    """from fractions import Fraction,gcd

a,b,c,d = [int(x) for x in input().split()]

if a*d > b*c:
    num = a*d-b*c
    denom = a*d
else:
    num = b*c-a*d
    denom = b*c
div = gcd(num,denom)
print('%d/%d'%(num//div,denom//div))"""
]

if __name__ == "__main__":
    for test in tests:
        print("ORIGINAL:", test, sep="\n\n")
        print("PATCHED:", patch_code(test, "Python 3"), sep="\n\n")
        print("=" * 50)