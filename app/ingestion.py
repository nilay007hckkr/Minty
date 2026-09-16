import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document


def load_and_chunk(data_dir: str = "sample_data") -> list[Document]:
    dir_loader = DirectoryLoader(
        path=data_dir,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = dir_loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    all_final_chunks = []

    for doc in documents:
        header_chunks = md_splitter.split_text(doc.page_content)

        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)

        final_chunks = recursive_splitter.split_documents(header_chunks)
        all_final_chunks.extend(final_chunks)

    for chunk in all_final_chunks:
        chunk.metadata["chunk_id"] = hashlib.sha256(
            chunk.page_content.encode()
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
