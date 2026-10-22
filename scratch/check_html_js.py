import os
import re

with open("static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

print("1. Checking style tags:")
has_open_style = "<style>" in html
has_close_style = "</style>" in html
print(f"  <style> present: {has_open_style}, </style> present: {has_close_style}")

print("\n2. Checking script tags:")
scripts = re.findall(r"<script(?:\s+[^>]*)?>(.*?)</script>", html, re.DOTALL)
print(f"  Number of script blocks: {len(scripts)}")

for idx, js in enumerate(scripts):
    if not js.strip():
        continue
    print(f"\n--- Script #{idx+1} ({len(js)} chars) ---")
    
    # Check variable declarations for duplicates in the top level
    const_vars = re.findall(r"^\s*const\s+([a-zA-Z0-9_$]+)\s*=", js, re.MULTILINE)
    let_vars = re.findall(r"^\s*let\s+([a-zA-Z0-9_$]+)\s*=", js, re.MULTILINE)
    func_names = re.findall(r"^\s*function\s+([a-zA-Z0-9_$]+)\s*\(", js, re.MULTILINE)
    
    seen = {}
    duplicates = []
    for v in const_vars + let_vars + func_names:
        if v in seen:
            duplicates.append(v)
        seen[v] = seen.get(v, 0) + 1
    
    if duplicates:
        print(f"  WARNING: Duplicate declarations found: {duplicates}")
    else:
        print("  No duplicate top-level declarations found.")
        
    # Check getElementById calls
    ids_in_js = set(re.findall(r"getElementById\([\"']([^\"']+)[\"']\)", js))
    ids_in_html = set(re.findall(r"id=[\"']([^\"']+)[\"']", html))
    
    missing = ids_in_js - ids_in_html
    print(f"  IDs referenced in JS: {len(ids_in_js)}")
    print(f"  IDs defined in HTML: {len(ids_in_html)}")
    if missing:
        print(f"  WARNING: IDs in JS not defined in HTML: {missing}")
    else:
        print("  All referenced IDs exist in HTML.")

print("\n3. Tag balance check:")
tags_to_check = ["html", "head", "body", "nav", "section", "div", "svg", "button", "table", "footer", "script", "style"]
for tag in tags_to_check:
    open_count = len(re.findall(rf"<{tag}(?:\s+[^>]*)?>", html, re.IGNORECASE))
    close_count = len(re.findall(rf"</{tag}>", html, re.IGNORECASE))
    status = "OK" if open_count == close_count else f"MISMATCH! (open={open_count}, close={close_count})"
    print(f"  <{tag}>: {status}")
