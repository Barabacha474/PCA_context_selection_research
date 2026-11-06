# Downstream LLM evaluation (exp 3.3)

Generator: claude-opus-5-5; judge: shupp; budget 8192; seed 0; tasks 6

## Absolute scores (mean over tasks, 1-10)

| method        |   idea_variety |   originality |   coherence |   overall |   n_used |   n_ctx |   answer_words |
|:--------------|---------------:|--------------:|------------:|----------:|---------:|--------:|---------------:|
| fps           |           7.33 |          7.5  |        8.5  |      7.83 |     1.33 |      36 |         338.5  |
| iterative_pca |           7.83 |          8.17 |        8.83 |      8.67 |     0.5  |      36 |         350.67 |
| none          |           7.83 |          7.67 |        9    |      8.67 |   nan    |       0 |         374.83 |
| random        |           7.83 |          8.17 |        8.83 |      8.67 |     0.83 |      33 |         391.5  |
| relevance     |           8    |          8    |        8.33 |      8.33 |     2.33 |      32 |         343.17 |

## Pairwise win rate (tie = 0.5)

|               |   win_rate |
|:--------------|-----------:|
| random        |      0.847 |
| relevance     |      0.236 |
| fps           |      0.222 |
| iterative_pca |      0.694 |

## Head-to-head (share for first-named)

|                            |   share_first |   n |
|:---------------------------|--------------:|----:|
| random vs relevance        |         0.958 |  12 |
| random vs fps              |         1     |  12 |
| random vs iterative_pca    |         0.583 |  12 |
| relevance vs fps           |         0.5   |  12 |
| relevance vs iterative_pca |         0.167 |  12 |
| fps vs iterative_pca       |         0.167 |  12 |

Position bias: first-presented response wins 48.61% of pairwise judgements.


API usage this run: generator {'prompt': 0, 'completion': 0, 'calls': 0, 'cached': 30}, judge {'prompt': 498546, 'completion': 4380, 'calls': 126, 'cached': 0}
