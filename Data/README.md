# Data Directory

This folder contains reference documents and legal databases used by the NyayShayak system.

## Contents

This folder should contain:
- Legal reference PDFs (IPC, BNS, etc.)
- Case law databases
- Training datasets (if any)

## Note for Git Users

The `Data/` folder is excluded from version control (.gitignored) because:
- Files can be large (PDFs, databases)
- May contain sensitive or copyrighted material
- Easy to download/add separately

## Setup

If you clone this repository, create a `Data/` folder and add your reference documents:

```powershell
mkdir Data
# Add your legal reference PDFs here
```

Example files you might add:
- `ipc_bns.pdf` - Indian Penal Code / Bharatiya Nyaya Sanhita reference
- Other legal reference documents as needed

These files will be used by the RAG system for legal analysis.
