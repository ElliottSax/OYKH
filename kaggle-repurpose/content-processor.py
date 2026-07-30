"""
RepurposeAI Content Processor for Kaggle
Process 100s of articles for FREE

Instead of: OpenAI API ($0.50-1.00 per article)
On Kaggle: $0 for unlimited processing
"""

from transformers import pipeline
import json

print("=" * 60)
print("RepurposeAI Content Processor")
print("FREE content transformation on Kaggle")
print("=" * 60)

# Load models (first time downloads, then cached)
print("\nLoading AI models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")
print("Models loaded!")

# Example: Process articles from your database
articles = [
    {
        "id": 1,
        "title": "10 Productivity Hacks",
        "content": "Long article content here..."
    },
    # ... 100 more articles
]

results = []

print(f"\nProcessing {len(articles)} articles...")

for article in articles:
    print(f"\n📄 Processing: {article['title']}")

    # Generate summary
    summary = summarizer(
        article['content'],
        max_length=150,
        min_length=50,
        do_sample=False
    )[0]['summary_text']

    # Analyze sentiment
    sentiment_result = sentiment(article['content'])[0]

    # Generate Twitter thread (simplified)
    twitter_thread = []
    sentences = article['content'].split('. ')
    for i in range(0, min(10, len(sentences)), 2):
        tweet = sentences[i] + '. ' + sentences[i+1] if i+1 < len(sentences) else sentences[i]
        twitter_thread.append(tweet[:280])  # Twitter limit

    result = {
        "id": article["id"],
        "title": article["title"],
        "summary": summary,
        "sentiment": sentiment_result,
        "twitter_thread": twitter_thread,
        "linkedin_post": f"{article['title']}\n\n{summary}\n\n#productivity #growth"
    }

    results.append(result)
    print(f"  ✅ Summary: {summary[:100]}...")
    print(f"  ✅ Sentiment: {sentiment_result['label']}")
    print(f"  ✅ Twitter thread: {len(twitter_thread)} tweets")

# Save results
output_path = "/kaggle/working/processed_content.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"✅ Processed {len(articles)} articles")
print(f"💾 Saved to: {output_path}")
print("=" * 60)
print(f"\nCost with OpenAI API: ${len(articles) * 0.75}")
print("Cost on Kaggle: $0")
print(f"Savings: ${len(articles) * 0.75}")
print("\nUpload results back to your database!")
