import hashlib
from pathlib import Path
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document


def load_and_chunk(data_dir: str = "sample_data") -> list[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    # strip_headers=False keeps the "## What documents do I need...?" line in
    # the chunk text. With headers stripped, the question a chunk answers only
    # lived in metadata, invisible to both the embedder and the cross-encoder.
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_final_chunks = []

    # sorted() so chunk order (and therefore the index) is deterministic.
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        header_chunks = md_splitter.split_text(text)

        for chunk in header_chunks:
            # as_posix() so sources are identical whether indexed on Windows
            # or in the Linux container.
            chunk.metadata["source"] = path.as_posix()

        final_chunks = recursive_splitter.split_documents(header_chunks)
        all_final_chunks.extend(final_chunks)

    for chunk in all_final_chunks:
        chunk.metadata["chunk_id"] = hashlib.sha256(
            f"{chunk.metadata['source']}\n{chunk.page_content}".encode()
        ).hexdigest()[:16]

    return all_final_chunks


if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Total chunks created: {len(chunks)}\n")
    for chunk in chunks:
        print(chunk.metadata)
        print(chunk.page_content)
        print(len(chunk.page_content))
        print("---")
