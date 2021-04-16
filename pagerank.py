import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory="./corpus0"):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus: dict, entry: str, damping_factor: float = DAMPING):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probDictionary = dict()
    randomPageProb = 0
    linkedPageProb = 0

    linkedPages = corpus[entry]

    # # For every page in corpus, check if dictionary contains the current link
    # for entry in corpus:
    #     links = corpus[entry]
    #     if page in links:
    #         linkedPages.append(entry)

    if len(linkedPages) > 0:
        linkedPageProb = damping_factor / len(linkedPages)
        randomPageProb =  (1 - damping_factor) / len(corpus)
    else:
        linkedPageProb = 0
        randomPageProb = 1 / len(corpus)    # Catch the edge case where the current page doesn't link to anything

    for entry in corpus:
        probDictionary[entry] = randomPageProb
        if entry in linkedPages:
            probDictionary[entry] += linkedPageProb

    return probDictionary


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ##set random seed

    # Initialize results dictionary
    pageRank = dict()

    # Initialize Corpus

    # Pick a starting location at random
    currentPage = random.choice(list(corpus))

    # iterate n times - adding to the value of pagerank each time
    for i in range(n):
        newLocations = transition_model(corpus, currentPage, damping_factor)
        choice = random.choices(list(newLocations.keys()), weights=list(newLocations.values()), k=1)[0]
        if choice in pageRank.keys():
            pageRank[choice] += 1
        else:
            pageRank[choice] = 1
        currentPage = choice

    # Find the normalizing factor (sum of page ranks)
    normFactor = sum(list(pageRank.values()))

    # Convert page rank to decimal - add any pages that are missing
    for page in corpus:
        if page in pageRank:
            pageRank[page] /= normFactor
        else:
            pageRank[page] = 0 # catch any pages that were never found in the sampling

    return  pageRank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.

    PageRank Formula: PR(p) = (1-damping_factor)/N + (damping_factor)SUM(PR(i)/links(i)) where i is each page that link to page p
    """

    #Initialize constants
    corpusLength = len(corpus)
    leadingTerm = (1-damping_factor) / corpusLength

    # initialize pageRank for all items
    pageRank = dict()
    for page in corpus:
        pageRank[page] = 1/corpusLength

    # Invert Corpus (transform from "A links to pages []" -> "A is linked to be pages {}"
    inverseCorpus = dict ()
    for page in corpus:
        inverseCorpus[page] = set()

    for page in corpus:
        # Catch case where page links to nothing (treat it as if it links to everywhere)
        if len(list(corpus[page])) == 0:
            for page2 in corpus:
                inverseCorpus[page2].add(page)

        # Otherwise add all links to inverse corpus
        else:
            for link in corpus[page]:
                inverseCorpus[link].add(page)
                # catch case where page links to nothing

    acceptedResidual = 0.001

    done = False
    interations = 0
    while done == False:
        done = True
        for page in pageRank:
            newPageRank = 0
            sumSeries = 0
            #Find all pages that link to page p add add their pageRank to the sum series
            for link in inverseCorpus[page]:
                sumSeries += pageRank[link]/len(list(corpus[link]))
            newPageRank = leadingTerm + damping_factor*sumSeries
            if max(newPageRank - pageRank[page], pageRank[page] - newPageRank) > 0.001:
                done = False
            pageRank[page] = newPageRank
            interations += 1
    for page in pageRank:
        pageRank[page] = round(pageRank[page], 4)

    return pageRank


if __name__ == "__main__":
    main()
