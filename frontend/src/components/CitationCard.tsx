import type { Citation } from "../types/api";

interface CitationCardProps {
  citation: Citation;
}

export function CitationCard({ citation }: CitationCardProps) {
  return (
    <div className="citation-card">
      <div className="citation-header">
        <span className="citation-id">{citation.citation_id}</span>
        <span className="citation-score">score: {citation.score.toFixed(3)}</span>
      </div>
      <div className="citation-source">
        {citation.source_path} | {citation.chunk_id}
      </div>
      <div className="citation-snippet">{citation.snippet}</div>
    </div>
  );
}
