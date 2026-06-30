# Paper Build Report

Date: 2026-06-30

## Command

```powershell
& "C:\Users\ADMIN\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" `
  -interaction=nonstopmode `
  -halt-on-error `
  main.tex
```

The command was run multiple times to resolve cross-references and citations.

## Result

```text
Output written on main.pdf (17 pages, 467831 bytes).
```

The rendered PDF therefore satisfies the FISAT/Springer full-paper page range
of 12-20 pages.

## Remaining Warnings

The final build has no undefined citation or undefined reference warnings.
There are minor underfull/overfull box warnings caused by long monospace command
paths in the experiment protocol. These do not stop PDF generation.
