from ai.githubai. groq_client import ask_groq

def chunk_code(code: str, chunk_size=8000) -> list[str]:
    lines = code.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def generate_doc_from_code(code: str) -> str:
    """Generates documentation by sending chunks of code to the AI."""
    if not code.strip():
        return "No code found in the repository to document."
        
    chunks = chunk_code(code)
    docs = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")

        # **ISSUE RESOLVED HERE**
        # The original prompt did not include the actual code `chunk`.
        # It also incorrectly used `len(chunk)` instead of `len(chunks)`.
        prompt = f"""
        You are an expert technical writer. Your task is to generate documentation from a piece of source code.

        Analyze the code provided below and generate structured documentation with the following sections:
        - **Purpose**: A high-level overview of what this code does.
        - **Important Classes/Functions**: A list of the main classes and functions, explaining what each one is for.
        - **Inline Logic Explanation**: Explain the code's logic step-by-step where necessary.

        This is chunk {i+1} of {len(chunks)} from a larger codebase. Focus only on the code within this chunk.

        Here is the code:
        ```
        {chunk}
        ```
        """
        
        try:
            doc_chunk = ask_groq(prompt)
            docs.append(f"{doc_chunk}")
        except Exception as e:
            # Handle potential API errors gracefully
            error_message = f"## Documentation for Chunk {i+1} / {len(chunks)}\n\n*Error processing this chunk: {str(e)}*"
            docs.append(error_message)

    return "\n\n---\n\n".join(docs)