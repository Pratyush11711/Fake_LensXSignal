import re

with open("static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

scripts = re.findall(r"<script(?:\s+[^>]*)?>(.*?)</script>", html, re.DOTALL)
js = scripts[1] if len(scripts) > 1 else scripts[0]

# Check if there are any duplicate declarations in the global scope
lines = js.split("\n")
brace_depth = 0
global_vars = {}

for line_num, line in enumerate(lines, 1):
    stripped = line.strip()
    
    if brace_depth == 0:
        m = re.match(r"^(const|let|var)\s+([a-zA-Z0-9_$]+)\s*=", stripped)
        if m:
            kind, name = m.groups()
            if name in global_vars:
                print(f"DUPLICATE GLOBAL VAR on line {line_num}: {kind} {name} (previously on line {global_vars[name]})")
            else:
                global_vars[name] = line_num

    # update brace depth
    brace_depth += line.count("{") - line.count("}")

print(f"Total top-level variables declared: {len(global_vars)}")
