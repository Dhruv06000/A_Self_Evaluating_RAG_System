from src.retrieval.contextual_retrieval import SelfRAG


def main():
    bot = SelfRAG()
    bot.start_chat()


if __name__ == "__main__":
    main()
