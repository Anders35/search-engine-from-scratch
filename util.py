class IdMap:
    """
    In practice, documents and terms are represented by integer IDs.
    This class maintains a two-way mapping between strings and integer IDs.
    """

    def __init__(self):
        """
        The mapping from string (term or document name) to ID is stored in a
        Python dictionary. The reverse mapping is stored in a Python list.

        Example:
            str_to_id["hello"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "hello"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Return the number of terms (or documents) stored in IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Return the string associated with index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Return the integer ID associated with string s.
        If s is not in IdMap, assign a new integer ID and return it.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def get_id_if_exists(self, s):
        """Return existing integer ID for string s, or None if absent."""
        return self.str_to_id.get(s)

    def contains(self, s):
        """Return True if string s already exists in the mapping."""
        return s in self.str_to_id

    def __getitem__(self, key):
        """
        __getitem__(...) is a Python special method that enables [] access
        syntax for custom collection classes.

        Reference:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        If key is an integer, use __get_str.
        If key is a string, use __get_id.
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

    def __contains__(self, key):
        """Support `in` checks for string keys."""
        if type(key) is str:
            return self.contains(key)
        return False


class TermFST:
    """A simple deterministic FST for mapping terms (strings) to integer IDs."""

    def __init__(self):
        self.transitions = [{}]
        self.outputs = [None]

    def add(self, term, output):
        """Add term with its output ID into the transducer."""
        state = 0
        for ch in term:
            next_state = self.transitions[state].get(ch)
            if next_state is None:
                next_state = len(self.transitions)
                self.transitions[state][ch] = next_state
                self.transitions.append({})
                self.outputs.append(None)
            state = next_state
        self.outputs[state] = output

    def transduce(self, term):
        """Return output ID for term, or None if term is not in the FST."""
        state = 0
        for ch in term:
            state = self.transitions[state].get(ch)
            if state is None:
                return None
        return self.outputs[state]


class FSTTermMap:
    """
    Term dictionary that stores string->ID transitions using a finite state
    transducer (FST), and keeps reverse ID->string mapping in a list.
    """

    def __init__(self):
        self.id_to_str = []
        self.fst = TermFST()

    def __len__(self):
        """Return the number of terms stored in FSTTermMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Return the term string associated with index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """Return existing term ID, or create a new ID and insert it to FST."""
        existing = self.fst.transduce(s)
        if existing is not None:
            return existing

        new_id = len(self.id_to_str)
        self.id_to_str.append(s)
        self.fst.add(s, new_id)
        return new_id

    def get_id_if_exists(self, s):
        """Return existing term ID for s, or None if absent."""
        return self.fst.transduce(s)

    def contains(self, s):
        """Return True if term s exists in the transducer."""
        return self.get_id_if_exists(s) is not None

    def __getitem__(self, key):
        """If key is int return term string, if str return/create term ID."""
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

    def __contains__(self, key):
        """Support `in` checks for term strings."""
        if type(key) is str:
            return self.contains(key)
        return False

    @classmethod
    def from_id_list(cls, id_to_str):
        """Build an FSTTermMap from ordered term list (ID order)."""
        term_map = cls()
        for term in id_to_str:
            term_map[term]
        return term_map

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merge two sorted lists of tuples (doc_id, tf) and return a sorted merged
    result. TF values are accumulated for matching doc IDs.

    Example: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
             posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

             return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                    = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Two sorted lists of tuples to be merged.

    Returns
    -------
    List[(Comparablem, int)]
        Sorted merged result.
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["hello", "everyone", "good", "morning", "everyone"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id is incorrect"
    assert term_id_map[1] == "everyone", "term_id is incorrect"
    assert term_id_map[0] == "hello", "term_id is incorrect"
    assert term_id_map["good"] == 2, "term_id is incorrect"
    assert term_id_map["morning"] == 3, "term_id is incorrect"

    fst_term_map = FSTTermMap()
    assert [fst_term_map[term] for term in doc] == [0, 1, 2, 3, 1], "fst term_id is incorrect"
    assert fst_term_map.get_id_if_exists("good") == 2, "fst lookup is incorrect"
    assert fst_term_map.get_id_if_exists("night") is None, "fst unknown lookup is incorrect"
    assert fst_term_map[1] == "everyone", "fst reverse lookup is incorrect"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id is incorrect"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs is incorrect"
