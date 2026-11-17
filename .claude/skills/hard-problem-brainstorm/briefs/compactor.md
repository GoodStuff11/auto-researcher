# Brief: Compactor

<!-- Given to a compaction sub-agent (ref/records.md, "Compaction"). A cheaper model is fine. -->

You make working files of a brainstorm in `[<wd>]` fit their size caps without losing anything a reader needs. You never change a verdict, a state, a Question or an Answer.

**Your context.** `python3 [<skill-dir>]/bs.py sizes [<wd>] --over` lists the files over cap. For a leaf, `bs.py context [<wd>] [P<id>] --role compactor` prints its summary, problem, one line per card and its log files with sizes.

**For each file over cap:**
1. Move superseded or historical text (old portfolio rows, round narratives, earlier statuses, pasted outputs) to the leaf's newest log (`leaves/<id>/log/round-NN.md`), or to `history.md` for the front page. Leave a one-line pointer only if a reader would look for it.
2. Replace repeated facts with a pointer to their one home: an idea's detail lives on its card, a lesson in `library.md`, a number in the log or card that produced it.
3. Shorten long field values (`Updated` is one line; certificate items are one or two sentences with ids).
4. Check that every id the file mentions still resolves (`bs.py show <id>`), that `check_brainstorm.py` reports no missing field, and that the file is under its cap.

Return a list of the files changed, their sizes before and after, and anything you could not shorten without losing content (the orchestrator decides).
