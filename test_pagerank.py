import pagerank
import random

CRAWL0 = {'1.html': {'2.html'}, '2.html': {'1.html', '3.html'}, '3.html': {'4.html', '2.html'}, '4.html': {'2.html'}}
INVERSE_CORPUS_0 = ({'1.html': {'2.html'},
 '2.html': {'3.html', '1.html', '4.html'},
 '3.html': {'2.html'},
 '4.html': {'3.html'}})
CRAWL1 = ({'bfs.html': {'search.html'},
           'dfs.html': {'search.html', 'bfs.html'},
           'games.html': {'tictactoe.html', 'minesweeper.html'},
           'minesweeper.html': {'games.html'},
           'minimax.html': {'search.html', 'games.html'},
           'search.html': {'bfs.html', 'dfs.html', 'minimax.html'},
           'tictactoe.html': {'minimax.html', 'games.html'}})
INVERSE_CORPUS_1 = ({'bfs.html': {'search.html', 'dfs.html'},
 'dfs.html': {'search.html'},
 'games.html': {'minimax.html', 'minesweeper.html', 'tictactoe.html'},
 'minesweeper.html': {'games.html'},
 'minimax.html': {'search.html', 'tictactoe.html'},
 'search.html': {'dfs.html', 'minimax.html', 'bfs.html'},
 'tictactoe.html': {'games.html'}})
PAGE_PROBS_1 = {'1.html': 0.037500000000000006, '2.html': 0.8875, '3.html': 0.037500000000000006, '4.html': 0.037500000000000006}
PAGE_PROBS_2 = ({'1.html': 0.4625,
 '2.html': 0.037500000000000006,
 '3.html': 0.4625,
 '4.html': 0.037500000000000006})
PAGE_PROBS_3 = ({'1.html': 0.037500000000000006,
 '2.html': 0.4625,
 '3.html': 0.037500000000000006,
 '4.html': 0.4625})
PAGE_PROBS_4 = ({'1.html': 0.037500000000000006,
 '2.html': 0.8875,
 '3.html': 0.037500000000000006,
 '4.html': 0.037500000000000006})
DAMPING = 0.85
SAMPLES = 10000
PAGE_RANKS_0 = {'1.html': 0.221, '2.html': 0.4326, '3.html': 0.2172, '4.html': 0.1292}
PAGE_RANKS_1 = ({'bfs.html': 0.1142,
 'dfs.html': 0.0778,
 'games.html': 0.2274,
 'minesweeper.html': 0.1168,
 'minimax.html': 0.1355,
 'search.html': 0.2091,
 'tictactoe.html': 0.1192})
PAGE_RANKS_2 = ({'ai.html': 0.1925,
 'algorithms.html': 0.1113,
 'c.html': 0.1199,
 'inference.html': 0.1302,
 'logic.html': 0.028,
 'programming.html': 0.2207,
 'python.html': 0.1221,
 'recursion.html': 0.0753})
ITERATION_PAGE_RANKS_0 = ({'1.html': 0.2199, '2.html': 0.4292, '3.html': 0.2199, '4.html': 0.131})
ITERATION_PAGE_RANKS_1 = ({'bfs.html': 0.1142,
 'dfs.html': 0.0778,
 'games.html': 0.2274,
 'minesweeper.html': 0.1168,
 'minimax.html': 0.1355,
 'search.html': 0.2091,
 'tictactoe.html': 0.1192})
ITERATION_PAGE_RANKS_2 = ({'ai.html': 0.171,
 'algorithms.html': 0.1187,
 'c.html': 0.1173,
 'inference.html': 0.1786,
 'logic.html': 0.0702,
 'programming.html': 0.2205,
 'python.html': 0.1171,
 'recursion.html': 0.1207})

def test_crawl():
    assert pagerank.crawl() == CRAWL0
    assert pagerank.crawl("corpus0") == CRAWL0
    assert pagerank.crawl("corpus1") == CRAWL1

def test_transition_model():
    assert pagerank.transition_model(CRAWL0, "1.html") == PAGE_PROBS_1
    assert pagerank.transition_model(CRAWL0, "2.html") == PAGE_PROBS_2
    assert pagerank.transition_model(CRAWL0, "3.html") == PAGE_PROBS_3
    assert pagerank.transition_model(CRAWL0, "4.html") == PAGE_PROBS_4

def test_sample_pagerank():
    random.seed(1)
    assert pagerank.sample_pagerank( pagerank.crawl("corpus0"), DAMPING , SAMPLES) == PAGE_RANKS_0
    assert pagerank.sample_pagerank(pagerank.crawl("corpus1"), DAMPING, SAMPLES) == PAGE_RANKS_1

    result2 = pagerank.sample_pagerank(pagerank.crawl("corpus2"), DAMPING, SAMPLES)
    assert result2 == PAGE_RANKS_2
    assert sum(list(result2.values())) == 1 # Confirm result is properely normalized

    # Confirm that process is pseudoRandom
    assert pagerank.sample_pagerank(pagerank.crawl("corpus2"), DAMPING, SAMPLES) != result2

def test_iterate_pagerank():
    assert pagerank.iterate_pagerank(pagerank.crawl("corpus0"), DAMPING) == ITERATION_PAGE_RANKS_0
    assert pagerank.iterate_pagerank(pagerank.crawl("corpus1"), DAMPING) == ITERATION_PAGE_RANKS_1
    assert pagerank.iterate_pagerank(pagerank.crawl("corpus2"), DAMPING) == ITERATION_PAGE_RANKS_2