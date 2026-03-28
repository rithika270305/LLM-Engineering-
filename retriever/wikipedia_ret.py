from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "The Last Ship TV series"
docs = retriever.invoke(query)

print("Number of docs:", len(docs))   

for i, doc in enumerate(docs, start=1):
    print(f"\nResult {i}")
    print("Title:", doc.metadata.get("title"))
    print("Content:", doc.page_content)