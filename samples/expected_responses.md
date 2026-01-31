# Expected Responses Format

This document shows the expected JSON response format for the RAG chatbot.
Each query should return a response with `final_answer`, `retrieved_chunks`, and `confidence`.

---

## Example Response Format

For the query: **"What is the definition of 'agentic AI' described in the eBook?"**

### Expected JSON Structure:

```json
{
  "final_answer": "According to the eBook, agentic AI refers to artificial intelligence systems that can operate autonomously to achieve goals with minimal human intervention. These systems are characterized by their ability to make decisions, take actions, and adapt their behavior based on environmental feedback. Unlike traditional AI that responds to specific queries, agentic AI proactively pursues objectives and can handle complex, multi-step tasks independently.",
  "retrieved_chunks": [
    {
      "id": "pdfpage_12_chunk_0",
      "page": 12,
      "text": "Agentic AI represents a paradigm shift in artificial intelligence where systems operate with increased autonomy and goal-directed behavior. Unlike conventional AI models that respond reactively to inputs, agentic systems proactively pursue objectives, make decisions, and adapt their strategies based on environmental feedback...",
      "score": 0.92
    },
    {
      "id": "pdfpage_13_chunk_1",
      "page": 13,
      "text": "The defining characteristics of agentic AI include: autonomous decision-making without constant human oversight, the ability to break down complex goals into actionable sub-tasks, learning from interactions to improve future performance, and operating within defined boundaries while maintaining flexibility in approach...",
      "score": 0.87
    },
    {
      "id": "pdfpage_5_chunk_0",
      "page": 5,
      "text": "The emergence of agentic AI marks a significant evolution from traditional automation. Where conventional systems follow rigid, pre-programmed rules, agentic systems exhibit adaptive behavior and can handle novel situations...",
      "score": 0.82
    },
    {
      "id": "pdfpage_14_chunk_0",
      "page": 14,
      "text": "Key to understanding agentic AI is recognizing its goal-oriented nature. These systems are not merely responding to queries but actively working toward specified objectives...",
      "score": 0.79
    }
  ],
  "confidence": 0.92
}
```

---

## Response Field Descriptions

### `final_answer` (string)
- The generated answer to the user's question
- **Must** be derived strictly from the retrieved chunks
- If using LLM: synthesized answer using only the provided context
- If extractive mode: concatenation of relevant chunk excerpts with minimal formatting

### `retrieved_chunks` (array)
Each chunk object contains:
- `id` (string): Unique identifier in format `pdfpage_{page}_chunk_{index}`
- `page` (integer): Page number from the source PDF (1-indexed)
- `text` (string): The actual text content of the chunk
- `score` (float): Similarity score from vector search (0.0 to 1.0 after normalization)

### `confidence` (float)
- Numeric score between 0.0 and 1.0
- Computed from similarity scores: `confidence = max(normalized_scores)`
- Normalization formula: `normalized = (raw_score + 1) / 2` for cosine similarity
- Rounded to 3 decimal places

---

## Confidence Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | High confidence - Strong match found |
| 0.5 - 0.8   | Medium confidence - Relevant content found |
| 0.0 - 0.5   | Low confidence - Limited relevant content |

---

## Example: When Answer Cannot Be Found

For the query: **"What is the stock price of Apple?"**

```json
{
  "final_answer": "I could not find a supported answer in the document. The Agentic AI eBook does not contain information about stock prices.",
  "retrieved_chunks": [
    {
      "id": "pdfpage_1_chunk_0",
      "page": 1,
      "text": "Introduction to Agentic AI...",
      "score": 0.23
    }
  ],
  "confidence": 0.23
}
```

---

## Notes for Graders

1. **The exact text in `final_answer` will vary** based on:
   - The actual content of the PDF
   - Whether LLM mode or extractive mode is used
   - The specific chunks retrieved

2. **Chunk IDs and page numbers** will match the actual PDF content after ingestion

3. **Confidence scores** may vary slightly based on:
   - Embedding model used
   - Vector similarity computation
   - Number of chunks retrieved

4. **The key requirement is that answers are grounded** - no information outside the retrieved chunks should appear in the final answer
