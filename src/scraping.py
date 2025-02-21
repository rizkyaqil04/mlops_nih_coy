import os
import re
import json

DATA_PATH = "data/raw/sinta_papers.json"

async def get_max_pages(query):
    schema = {
        "name": "Pagination Info",
        "baseSelector": "div.text-center.pagination-text",
        "fields": [
            {"name": "pagination", "selector": "small", "type": "text"}
        ]
    }

    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
        )

        url = f"https://sinta.kemdikbud.go.id/google?q={query.replace(' ', '+')}"

        result = await crawler.arun(url=url, config=config)
        pagination_text = json.loads(result.extracted_content)

        if pagination_text:
            match = re.search(r"Page \d+ of (\d+)", pagination_text[0]["pagination"])
            if match:
                return int(match.group(1))

    return 1  # Default jika tidak ditemukan

async def scrape_sinta(query):
    max_pages = await get_max_pages(query)

    schema = {
        "name": "Sinta Papers",
        "baseSelector": "div.ar-list-item",
        "fields": [
            {"name": "title", "selector": "div.ar-title", "type": "text"},
            {"name": "description", "selector": "div.ar-meta", "type": "text"}
        ]
    }

    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)
    all_papers = []

    async with AsyncWebCrawler() as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
        )

        for page in range(1, max_pages + 1):
            url = f"https://sinta.kemdikbud.go.id/google?page={page}&q={query.replace(' ', '+')}"
            result = await crawler.arun(url=url, config=config)
            papers = json.loads(result.extracted_content)
            all_papers.extend(papers)

    # Ensure the 'data' directory exists
    os.makedirs("data/raw", exist_ok=True)

    # Save to JSON
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=4)

    print(f"Scraped data saved to {DATA_PATH}")

    return all_papers

if __name__ == "__main__":
    import asyncio
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

    query = "machine learning"
    asyncio.run(scrape_sinta(query))

