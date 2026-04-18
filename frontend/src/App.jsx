import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [searchedQuery, setSearchedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  async function handleSearch(event) {
    event.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError("Please enter a search query.")
      setResults([])
      setHasSearched(false)
      return;
    }

    setLoading(true);
    setError("");
    setHasSearched(true);
    setSearchedQuery(trimmedQuery);

    try {
      const response = await fetch(
        `/api/search?q=${encodeURIComponent(trimmedQuery)}&top_k=10`
      );

      if (!response.ok) {
        throw new Error (`Search request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || "Something went wrong while searchign");
      setResults([]);
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Research Paper Search</h1>
        <p>
          Keyword search over research paper titles and abstracts
          using BM25.
        </p>
      </header>

      <main className="main-content">
        <form className="search-form" onSubmit={handleSearch}>
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search for topics like transformer, graph neural network, reinforcement learning..."
            className="search-input"
          />
          <button type="submit" className="search-button" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {error && <div className="message error-message">{error}</div>}

        {hasSearched && !loading && !error && (
          <div className="results-header">
            <h2>Results</h2>
            <p>
              Showing {results.length} result{results.length !== 1 ? "s" : ""}{" "}
              for <strong>{searchedQuery}</strong>
            </p>
          </div>
        )}

        {hasSearched && !loading && !error && results.length === 0 && (
          <div className="message empty-message">
            No results found for <strong>{searchedQuery}</strong>.
          </div>
        )}

        <section className="results-list">
          {results.map((result) => (
            <article className="result-card" key={result.id}>
              <div className="result-top">
                <h3>{result.title}</h3>
                <span className="score-badge">
                  Score: {result.score.toFixed(2)}
                </span>
              </div>

              <div className="meta">
                {result.authors && <p><strong>Authors:</strong> {result.authors}</p>}
                <p>
                  <strong>Venue:</strong> {result.venue || "Unknown"} |{" "}
                  <strong>Year:</strong> {result.year ?? "Unknown"} |{" "}
                  <strong>Citations:</strong> {result.n_citation ?? 0}
                </p>
              </div>

              <p className="abstract">{result.abstract}</p>
            </article>
          ))}
        </section>
      </main>
    </div>
  );
}

