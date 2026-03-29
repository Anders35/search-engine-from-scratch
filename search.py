import argparse

from bsbi import BSBIIndex, get_postings_encoding


def parse_args():
    parser = argparse.ArgumentParser(description='Search dokumen dengan BSBI index')
    parser.add_argument('--compression', default='elias-gamma',
                        choices=['standard', 'vbe', 'elias-gamma'],
                        help='Jenis kompresi postings yang dipakai saat membaca index')
    parser.add_argument('--scoring', default='bm25', choices=['tfidf', 'bm25'],
                        help='Skema scoring retrieval')
    parser.add_argument('-k', type=int, default=10, help='Top-K dokumen')
    return parser.parse_args()


def main():
    args = parse_args()

    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = get_postings_encoding(args.compression), \
                              output_dir = 'index')

    queries = ["alkylated with radioactive iodoacetate", \
               "psychodrama for disturbed children", \
               "lipid metabolism in toxemia and normal pregnancy"]
               
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve(query, k = args.k, scoring = args.scoring):
            print(f"{doc:30} {score:>.3f}")
        print()


if __name__ == '__main__':
    main()