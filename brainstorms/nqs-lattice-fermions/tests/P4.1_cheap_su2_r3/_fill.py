import sys
s = open("_section.txt").read()
a = s.index("```\n4x2 cylinder") + 4
b = s.index("```", a)
s = s[:a] + open("logs/smoke_output.txt").read() + s[b:]
if len(sys.argv) > 1:
    s = s.replace("PRECHECK_PLACEHOLDER", open(sys.argv[1]).read().strip())
open("_section.txt", "w").write(s)
