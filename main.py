from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from vectors import create_retriever, prompt


def extract_video_id(user_input: str) -> str:
    user_input = user_input.strip()
    if "youtube.com/watch" in user_input:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(user_input)
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    elif "youtu.be/" in user_input:
        return user_input.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    return user_input


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def main():
    model = OllamaLLM(model="llama3.2")
    parser = StrOutputParser()
    quit_commands = {"q", "quit", "exit", "quiet", "q!"}

    print("=== YouTube RAG Chatbot ===")

    while True:
        raw_input = input("\nEnter YouTube Video ID (or 'q' to quit): ").strip()

        if not raw_input or raw_input.lower() in quit_commands:
            print("Exiting YouTube RAG Chatbot. Goodbye!")
            break

        video_id = extract_video_id(raw_input)

        try:
            retriever = create_retriever(video_id)
        except Exception as e:
            print(f"\n[Error]: {e}")
            print("Redirecting back to Video ID input...\n")
            continue

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
        })

        chain = parallel_chain | prompt | model | parser

        print(f"\n--- Chatting with Video ID: {video_id} ---")
        print("Type 'q', 'quit', 'exit', or 'quiet' to switch video / quit asking.\n")

        while True:
            question = input("Ask a question: ").strip()

            if question.lower() in quit_commands:
                print("\nReturning to YouTube Video ID prompt...")
                break

            if not question:
                continue

            try:
                result = chain.invoke(question)
                print(f"\n{result}\n")
            except Exception as e:
                print(f"\n[Error generating answer]: {e}\n")


if __name__ == "__main__":
    main()