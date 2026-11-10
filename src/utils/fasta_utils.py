"""Read FASTA sequences or headers for clustering and alignment.

`read_fasta` uses the first whitespace-separated token of each header as the ID.
Clustering and alignment inputs use sequence hashes as these IDs.

`read_fasta` returns the sequences. `read_fasta_hashes` returns only the headers in file
order, which is what connected-component clustering needs, including sequences with no
neighbour above the identity threshold that must stay singletons.

"""
from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    """Read FASTA records keyed by the first token of each header.

    Multi-line sequences are joined. Blank lines are skipped.

    Args:
        path: FASTA file holding one or more records.

    Returns:
        Sequences keyed by their FASTA IDs, in file order.

    Raises:
        ValueError: A record is empty, its ID is repeated or absent, a sequence appears
            before any header, or the file holds no records.
    """
    records: dict[str, str] = {}
    current_id = None
    chunks: list[str] = []

    def close_record() -> None:
        sequence = ''.join(chunks)
        if not sequence:
            raise ValueError(f'Empty FASTA sequence for {current_id} in {path}.')
        records[current_id] = sequence

    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if current_id is not None:
                close_record()
            tokens = line[1:].split()
            current_id = tokens[0] if tokens else None
            if current_id is None or current_id in records:
                raise ValueError(f'Empty or repeated FASTA ID in {path}.')
            chunks = []
        elif line.strip():
            if current_id is None:
                raise ValueError(f'FASTA sequence before its header in {path}.')
            chunks.append(line.strip())

    if current_id is not None:
        close_record()
    if not records:
        raise ValueError(f'No FASTA records in {path}.')
    return records


def read_fasta_hashes(fasta_path: Path) -> list:
    """Return the record headers (= sequence hashes) of a FASTA, in file order.

    `export_function_fasta` writes one record per unique sequence with the hash as the
    header, so this is the exact node set for connected-component clustering, including
    sequences with no `>= t` neighbour, which must stay singletons.

    Reads headers only, so it does not check the sequences. Use `read_fasta` when the
    sequences are needed.

    Args:
        fasta_path: FASTA file whose headers are sequence hashes.

    Returns:
        The headers, in file order.
    """
    hashes = []
    with Path(fasta_path).open() as f:
        for line in f:
            if line.startswith('>'):
                hashes.append(line[1:].strip())
    return hashes
