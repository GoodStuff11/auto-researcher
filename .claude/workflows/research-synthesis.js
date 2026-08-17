// .claude/workflows/research-synthesis.js
export const meta = {
  name: 'research-synthesis',
  description: 'Score, read, and synthesize a candidate paper pool into an answer to a research question',
  phases: [
    { title: 'Score' },
    { title: 'Read' },
    { title: 'Synthesize' },
  ],
}

const { question, candidates, maxRead = 50 } = args

phase('Score')

const BATCH_SIZE = 10
const batches = []
for (let i = 0; i < candidates.length; i += BATCH_SIZE) {
  batches.push(candidates.slice(i, i + BATCH_SIZE))
}

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          relevance: { type: 'number' },
          reason: { type: 'string' },
        },
        required: ['id', 'relevance'],
      },
    },
  },
  required: ['scores'],
}

const scoredBatches = await parallel(
  batches.map(batch => () =>
    agent(
      `Question: "${question}"\n\nScore each paper's relevance to this question from 0 ` +
      `(irrelevant) to 10 (directly on-topic). Consider tangential/adjacent-field relevance ` +
      `too, not just exact keyword matches.\n\nPapers:\n` +
      JSON.stringify(batch.map(p => ({ id: p.id, title: p.title, abstract: p.abstract })), null, 2),
      { phase: 'Score', schema: SCORE_SCHEMA }
    )
  )
)

const scoreById = {}
for (const result of scoredBatches.filter(Boolean)) {
  for (const s of result.scores) {
    scoreById[s.id] = s
  }
}

const qualified = candidates
  .filter(p => scoreById[p.id] && scoreById[p.id].relevance >= 5)
  .sort((a, b) => scoreById[b.id].relevance - scoreById[a.id].relevance)

const ranked = qualified.slice(0, maxRead)

if (qualified.length > ranked.length) {
  log(
    `${ranked.length} of ${qualified.length} papers scoring >=5/10 relevance were read in depth ` +
    `(capped at maxRead=${maxRead}); ${qualified.length - ranked.length} scored relevant but were ` +
    `NOT read this pass — report this gap, don't drop it silently.`
  )
} else {
  log(`${ranked.length} papers selected out of ${candidates.length} candidates`)
}

phase('Read')

const EXTRACT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    summary: { type: 'string' },
    approach: { type: 'string' },
    relation_to_question: { type: 'string' },
  },
  required: ['id', 'summary', 'relation_to_question'],
}

const extracts = await pipeline(
  ranked,
  paper =>
    agent(
      `Question: "${question}"\n\nRead this paper and extract what's relevant to the question.\n\n` +
      `Title: ${paper.title}\nAbstract: ${paper.abstract || '(no abstract available)'}\n` +
      `${paper.full_text_excerpt ? `Full text excerpt:\n${paper.full_text_excerpt}` : '(no full text available)'}\n\n` +
      `Return a summary of what the paper does, its approach/method, and specifically how it ` +
      `relates to the question (does it answer it, partially address it, or is it just adjacent).`,
      { phase: 'Read', schema: EXTRACT_SCHEMA, label: `read:${paper.id}` }
    ).then(extract => ({ ...extract, paper }))
)

phase('Synthesize')

const validExtracts = extracts.filter(Boolean)

function paperLink(paper) {
  if (paper.oa_pdf_url) return paper.oa_pdf_url
  if (paper.doi) return `https://doi.org/${paper.doi}`
  if (paper.landing_url) return paper.landing_url
  if (paper.arxiv_id) return `https://arxiv.org/abs/${paper.arxiv_id}`
  return null
}

// Citation numbers are assigned here, in ranked (relevance) order, and are
// the single source of truth for numbering — the report step reuses this
// exact list so inline [n] markers line up with the bibliography.
const citations = validExtracts.map((e, i) => ({
  n: i + 1,
  id: e.id,
  title: e.paper.title,
  authors: e.paper.authors,
  year: e.paper.year,
  venue: e.paper.venue,
  link: paperLink(e.paper),
}))

const citationKey = citations
  .map(c => `[${c.n}] id=${c.id} — ${(c.authors || []).slice(0, 3).join(', ')} (${c.year || 'n.d.'}) — link: ${c.link || '(no link available)'}`)
  .join('\n')

const synthesis = await agent(
  `Question: "${question}"\n\nBased on these paper extracts, write a synthesis answering the ` +
  `question directly. Structure your response as:\n\n` +
  `1. A direct answer (has this been done? fully / partially / not found) with a confidence ` +
  `level and why.\n` +
  `2. What's been done, organized by approach or theme.\n` +
  `3. Explicit gaps: what the question asks that nothing in this set addresses.\n\n` +
  `Citation rule: every claim that draws on a specific paper below must carry an inline citation ` +
  `right after the claim, formatted as a markdown link whose visible text is the bracketed number ` +
  `and whose target is that paper's link, e.g. "the J1-J2 model is hard to learn [3](${citations[0]?.link || 'https://example.com'})". ` +
  `Use the exact number and link given for that paper in the citation key below — do not invent ` +
  `numbers or links, and if a paper has "(no link available)" just cite the bare number "[n]" with ` +
  `no link for that one. A claim can carry multiple citations, e.g. "...[2](...)[5](...)". Do not ` +
  `add a references/bibliography section yourself — that is appended separately after your text.\n\n` +
  `Citation key (number → paper, for your reference only, do not print this list):\n${citationKey}\n\n` +
  `Extracts:\n` +
  JSON.stringify(
    validExtracts.map((e, i) => ({
      n: i + 1, id: e.id, summary: e.summary, approach: e.approach, relation: e.relation_to_question,
    })),
    null, 2
  ),
  { phase: 'Synthesize' }
)

return {
  synthesis,
  extracts: validExtracts,
  scores: Object.values(scoreById),
  citations,
  totalCandidates: candidates.length,
  totalRanked: ranked.length,
  totalQualified: qualified.length,
  unreadQualifiedIds: qualified.slice(ranked.length).map(p => p.id),
}
