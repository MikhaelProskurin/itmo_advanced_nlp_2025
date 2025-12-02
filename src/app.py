import asyncio
from agent.kinoteatr_ru_agent import KinoteatrRuAgent


async def main(question: str) -> str:

    agent = KinoteatrRuAgent()
    app = agent.compiled_graph

    return await app.ainvoke({"question": question})

if __name__ == "__main__":

    question = "<your_question>"
    answer = asyncio.run(main(question))

    print(answer["answer"])