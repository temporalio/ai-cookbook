import asyncio

from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    resp = await client.responses.create(
        model="gpt-5-mini",
        instructions="You only respond in haikus.",
        input="Tell me about recursion in programming",
    )

    print(resp.output_text)

if __name__ == "__main__":
    asyncio.run(main())