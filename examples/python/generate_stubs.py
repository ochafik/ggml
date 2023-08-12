"""
  This generates bindings for the ggml library using cffi and .pyi stubs for the Python bindings.

  See the various environment variables at the top of this file for options.
"""

import re
from sys import argv

def to_type(t):
  """
    from a C type, returns the corresponding Python type, or None if unknown.
  """
  t = t.strip()
  if t.endswith('*'):
    return 'ffi.CData'
  elif t.startswith('struct '):
    return 'ffi.CData'
  elif t.startswith('enum ') or t in ['int', 'int64_t', 'size_t', 'int32_t', 'int16_t', 'int8_t', 'uint64_t', 'uint32_t', 'uint16_t', 'uint8_t']:
    return 'int'
  elif t in ['float', 'double']:
    return 'float'
  elif t == 'void':
    return 'None'
  elif t == 'bool':
    return 'bool'
  print('## Unknown type:', t)
  return None

def format_arg(t, n):
  t = to_type(t)
  return n if t is None else f'{n}: {t}'

def format_ret(t):
  t = to_type(t)
  return '' if t is None else f' -> {t}'

def generate_stubs(header: str):
    """
      Hack to generates a .pyi Python stub file for the GGML API using C header files.

      The idea is to remove comments, then unbreak functions declared across multiple lines,
      then pseudo-parse the function name, return type, args and their types. Simple yet efficient.

      I might contribute a cleaner version of this to cffi at some point, using the
      actual AST it already has.
    """
    # Remove comments and ~ensure each GGML_API function signature is on a single line
    header = re.sub(r'/\*.*?\*/', '', header, flags=re.M | re.S)
    header = re.sub(r'//.*([\n\r])', '\\1', header)
    header = re.sub('([(,])\\s*[\\n\\r]\\s*', r'\1', header, flags=re.M | re.S)
    header = re.sub(r'[ \t]+', ' ', header)
    header = re.sub(r'\(\s*void\s*\)', '()', header)

    sigs = list(set([l.strip() for l in header.splitlines() if 'GGML_API' in l and 'GGML_DEPRECATED' not in l]))
    sigs.sort()

    # The cffi binding will expose an object `lib` with all the constants and functions.
    # We stub it using a Python class with static methods, which works w/ at least pylance / VS Code.
    lib = ['class lib:']
    for sig in sigs:
        m = re.search(r'GGML_API\s*(.*?)\b(\w+)\s*\(([^)]*)\)\s*;', sig)
        if m is not None:
            try:
                (rettype, name, arglist) = m.groups()
                arglist = [a.strip() for a in arglist.split(',')]
                
                args = []
                for arg in arglist:
                    if arg.strip() == '':
                        continue
                    if arg == '...':
                        args.append('*args')
                    else:
                        am = re.search(r'^(.+?)\b(\w+)$', arg)
                        args.append(format_arg(am.group(1), am.group(2)))
                lib += [f'  # {sig}', 
                        f'  def {name}({", ".join(args)}){format_ret(rettype)}: ...',
                        '']
            except:
                print('## Error parsing:', sig)
                raise
    return '\n'.join(lib)
