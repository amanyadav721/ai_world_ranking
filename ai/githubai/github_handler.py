import os
import tempfile
from git import Repo

def clone_repo(repo_url: str) -> str:
    """Clones a public GitHub repository to a temporary directory."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="repo_")
        Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except Exception as e:
        # Catch potential Git errors (e.g., repo not found, authentication required)
        raise ConnectionError(f"Failed to clone repository: {e}")


def extract_code_from_repo(repo_path: str) -> str:
    """
    Extracts code from supported file types in a repository path,
    concatenating them into a single string.
    """
    code = ""
    supported_extensions = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rb", ".go", ".rs")
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(supported_extensions):
                full_path = os.path.join(root, file)
                # Skip if file is very small (likely empty or trivial)
                if os.path.getsize(full_path) < 15:
                    continue
                
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()
                    # Skip if file is only whitespace
                    if file_content.strip():
                        # **FIX:** Use relative path for cleaner output
                        relative_path = os.path.relpath(full_path, repo_path)
                        code += f"\n\n--- \n# File: {relative_path}\n---\n{file_content}"
                except Exception:
                    # Ignore files that cannot be read
                    continue
    return code