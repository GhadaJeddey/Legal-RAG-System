"""
Article-level semantic chunker.

Strategy:
1. Structural Split : split by article-level + recording metadata per article ( section , title , chapter ...)
2. Semantic Split : If article is longer than a threshholt , it is split semantically 
"""
import re

from .base import Chunker

HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)$')
ARTICLE_RE = re.compile(r'\*{0,2}(Art\.?|Article)\s+([\d\-]+)', re.IGNORECASE)
TABLE_LINE_RE = re.compile(r'^\s*\|')

# French-aware sentence splitting: avoid breaking on abbreviations like
# "Art.", "n°", "M.", decimal numbers, etc.
_ABBREVIATIONS = ["Art", "Art", "n", "N", "M", "Mme", "MM", "art", "cf", "etc", "p", "art"]
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<!\b' + r')(?<!\b'.join(_ABBREVIATIONS) + r')(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9«])'
)

DEFAULT_MAX_TOKENS = 350  # articles above this get sub-split
MIN_FRAGMENT_TOKENS = 50
BREAKPOINT_PERCENTILE = 90


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_units(body: str) -> list[str]:
    """Split an article body into ordered units: sentences, with any
    markdown table kept as a single atomic unit."""
    lines = body.split("\n")
    units: list[str] = []
    buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_prose():
        if buffer:
            prose = "\n".join(buffer).strip()
            if prose:
                units.extend(_split_sentences(prose))
            buffer.clear()

    def flush_table():
        if table_buffer:
            units.append("\n".join(table_buffer))
            table_buffer.clear()

    in_table = False
    for line in lines:
        if TABLE_LINE_RE.match(line):
            if not in_table:
                flush_prose()
                in_table = True
            table_buffer.append(line)
        else:
            if in_table:
                flush_table()
                in_table = False
            buffer.append(line)
    flush_table()
    flush_prose()
    return units


def _parse_articles(markdown: str) -> list[dict]:
    """Segment the document into article-level blocks with breadcrumb metadata."""
    lines = markdown.split("\n")
    section_path_by_level: dict[int, str] = {}
    articles = []
    current = None

    def is_article_header(text: str) -> bool:
        return bool(ARTICLE_RE.search(text))

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip(" *")

            if is_article_header(title):
                if current is not None:
                    articles.append(current)
                article_num_match = ARTICLE_RE.search(title)
                current = {
                    "article_number": article_num_match.group(2) if article_num_match else None,
                    "title": title,
                    "breadcrumb": dict(sorted(section_path_by_level.items())),
                    "body_lines": [],
                }
            else:
                if current is not None:
                    articles.append(current)
                    current = None
                section_path_by_level[level] = title
                for deeper in list(section_path_by_level):
                    if deeper > level:
                        del section_path_by_level[deeper]
        else:
            if current is not None:
                current["body_lines"].append(line)

    if current is not None:
        articles.append(current)

    for a in articles:
        a["body"] = "\n".join(a.pop("body_lines")).strip()

    return articles


class ArticleChunker(Chunker):
    def __init__(
        self,
        embedding_model,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_fragment_tokens: int = MIN_FRAGMENT_TOKENS,
        breakpoint_percentile: int = BREAKPOINT_PERCENTILE,
    ):
        self.model = embedding_model
        self.max_tokens = max_tokens
        self.min_fragment_tokens = min_fragment_tokens
        self.breakpoint_percentile = breakpoint_percentile

    def _count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def chunk(self, markdown: str) -> list[dict]:
        articles = _parse_articles(markdown)
        chunks = []
        
        for article in articles:
            base_meta = {
                **{f"level_{lvl}": title for lvl, title in article["breadcrumb"].items()},
                "article_number": article["article_number"],
                "article_title": article["title"],
            }
            if not article["body"]:
                continue

            token_count = self._count_tokens(article["body"])

            if token_count <= self.max_tokens:
                chunks.append({
                    "text": article["body"],
                    "metadata": {
                        **base_meta,
                        "sub_chunk": None,
                        "sub_chunk_total": None,
                        "token_count": token_count,
                    },
                })
                continue

            sub_chunks = self._semantic_sub_split(article["body"])
            for i, sub_text in enumerate(sub_chunks):
                chunks.append({
                    "text": sub_text,
                    "metadata": {
                        **base_meta,
                        "sub_chunk": i,
                        "sub_chunk_total": len(sub_chunks),
                        "token_count": self._count_tokens(sub_text),
                    },
                })
        return chunks

    def _semantic_sub_split(self, body: str) -> list[str]:
        import numpy as np

        units = _split_into_units(body)
        if len(units) <= 1:
            return [body]

        embeddings = self.model.encode(units, normalize_embeddings=True, show_progress_bar=False)
        distances = [
            1 - float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        threshold = float(np.percentile(distances, self.breakpoint_percentile)) if distances else 0.0

        groups: list[list[str]] = [[units[0]]]
        boundary_distances: list[float] = []
        for i, d in enumerate(distances):
            if d > threshold:
                groups.append([units[i + 1]])
                boundary_distances.append(d)
            else:
                groups[-1].append(units[i + 1])

        merged = self._merge_small_groups(groups, boundary_distances)
        return ["\n".join(g).strip() for g in merged]

    def _merge_small_groups(
        self, groups: list[list[str]], boundary_distances: list[float]
    ) -> list[list[str]]:
        """Fold undersized groups into whichever neighbor is semantically
        closer (smaller cosine distance at that boundary), never exceeding
        max_tokens. Falls back to the other neighbor, or leaves the group
        standalone, when the closer merge would blow the token ceiling."""
        groups = [list(g) for g in groups]
        boundary_distances = list(boundary_distances)

        def tokens_of(*parts: list[str]) -> int:
            text = "\n".join(p for part in parts for p in part)
            return self._count_tokens(text)

        def fits(a: int, b: int) -> bool:
            return tokens_of(groups[a], groups[b]) <= self.max_tokens

        i = 0
        while i < len(groups):
            if self._count_tokens("\n".join(groups[i])) >= self.min_fragment_tokens:
                i += 1
                continue

            has_left = i > 0
            has_right = i < len(groups) - 1
            merge_left = has_left and fits(i - 1, i)
            merge_right = has_right and fits(i, i + 1)

            if merge_left and (not merge_right or boundary_distances[i - 1] <= boundary_distances[i]):
                groups[i - 1].extend(groups[i])
                del groups[i]
                del boundary_distances[i - 1]
                i -= 1
            elif merge_right:
                groups[i].extend(groups[i + 1])
                del groups[i + 1]
                del boundary_distances[i]
            else:
                # neither neighbor can absorb this group without exceeding
                # max_tokens; leave it standalone even though it's small.
                i += 1
        return groups
