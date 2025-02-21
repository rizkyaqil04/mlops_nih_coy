import asyncio
import json
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

SCRAPED_FILE = RAW_DIR / "sinta_scraped.json"
PREPROCESSED_FILE = PROCESSED_DIR / "sinta_preprocessed.json"
CLUSTERING_FILE = RESULTS_DIR / "clustering_results.json"
MODEL_PATH = MODELS_DIR / "bertopic_model"

async def main():
    from src.scraping import scrape_sinta
    from src.preprocessing import preprocess_papers
    from src.bert import compute_topics_with_bertopic

    query = "pomegranate"
    max_pages = 5

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ===============================================================================================
    # 🔍 Checking for existing scraped data
    if SCRAPED_FILE.exists():
        logging.info("✅ Scraped data found!")
        rescrape = input("🔄 Rescrape data? (y/n): ").strip().lower()
        if rescrape == "y":
            logging.info("🚀 Rescraping...")
            papers = await scrape_sinta(query)
            SCRAPED_FILE.write_text(json.dumps(papers, indent=2, ensure_ascii=False))
        else:
            logging.info("➡ Using existing scraped data.")
            papers = json.loads(SCRAPED_FILE.read_text(encoding="utf-8"))
    else:
        logging.info("❌ Scraped data not found! Scraping now...")
        papers = await scrape_sinta(query)
        SCRAPED_FILE.write_text(json.dumps(papers, indent=2, ensure_ascii=False))
    logging.info("✅ Scraping complete!")

    # ===============================================================================================
    # 🔍 Checking for existing preprocessed data
    if PREPROCESSED_FILE.exists():
        logging.info("✅ Preprocessed data found!")
        reprocess = input("🔄 Reprocess data? (y/n): ").strip().lower()
        if reprocess == "y":
            logging.info("🚀 Reprocessing...")
            cleaned_papers = preprocess_papers(papers)
            PREPROCESSED_FILE.write_text(json.dumps(cleaned_papers, indent=2, ensure_ascii=False))
        else:
            logging.info("➡ Using existing preprocessed data.")
            cleaned_papers = json.loads(PREPROCESSED_FILE.read_text(encoding="utf-8"))
    else:
        logging.info("❌ Preprocessed data not found! Preprocessing now...")
        cleaned_papers = preprocess_papers(papers)
        PREPROCESSED_FILE.write_text(json.dumps(cleaned_papers, indent=2, ensure_ascii=False))
    logging.info("✅ Preprocessing complete!")

    # ===============================================================================================
    # 🔍 Validating preprocessed data before clustering
    if not cleaned_papers:
        logging.error("❌ No valid preprocessed data found! Stopping process.")
        return

    logging.info("🔄 Running BERTopic for clustering...")
    try:
        topic_model, topics = compute_topics_with_bertopic(cleaned_papers)
        topic_info = topic_model.get_topic_info()

    except Exception as e:
        logging.error(f"❌ Error in BERTopic processing: {e}")
        return

    results = {"clusters": topics, "topic_info": topic_info.to_dict(orient="records")}
    CLUSTERING_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    logging.info(f"✅ Clustering complete! Results saved to '{CLUSTERING_FILE}'")

    # ===============================================================================================
    # 🔍 Displaying topic count and list
    num_topics = len(topic_info)
    topic_list = topic_info["Name"].tolist()

    logging.info(f"\n📊 Total Topics Found: {num_topics}")
    logging.info("📌 Topic List:")
    for idx, topic in enumerate(topic_list, start=1):
        logging.info(f"  {idx}. {topic}")

if __name__ == "__main__":
    asyncio.run(main())

